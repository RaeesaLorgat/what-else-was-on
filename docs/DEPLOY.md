# Deploying to Azure Container Apps

The replay page and the agent team run as one container on Azure Container Apps.

Azure for Students subscriptions are not allowed to build images in Azure (ACR Tasks), so the
image is built by GitHub Actions instead (`.github/workflows/container.yml`): on every push to
`main` it runs the tests, builds the `Dockerfile` and publishes the image to GitHub Container
Registry (`ghcr.io/raeesalorgat/what-else-was-on`). Container Apps runs that public image. The
Foundry key is stored as a Container Apps secret, never in the image or the repo.

What goes into the image: `src/`, `scripts/`, `web/` and `data/` from the repo (the agents' saved
stories are committed). During the build, matches 1-3 are generated (the same seeds always give the
same matches) and Karun Singh's xT grid is downloaded from his site. `.dockerignore` keeps `.env`,
`.git`, `.venv` and `tests/` out.

All commands are for PowerShell in VS Code, from the project folder.

## 1. One-time setup

```powershell
winget install -e --id Microsoft.AzureCLI
```

Close and reopen the VS Code terminal, then:

```powershell
az login
az account show -o table
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
```

`az login` opens the browser; pick the account with your Azure for Students subscription.

## 2. Choose a region and make a resource group for the app

Use the region your Foundry resource is in (Azure for Students only allows some regions):

```powershell
az cognitiveservices account list --query "[].{name:name, group:resourceGroup, region:location}" -o table
```

Make a separate resource group for the app, so it can be deleted later without touching Foundry
(replace `<REGION>` with the region from the table, for example `eastus2`):

```powershell
$REGION = "<REGION>"
$RG = "whatelse-app"
az group create --name $RG --location $REGION
```

## 3. Build the image on GitHub

Commit and push (the stories in `data/` are included):

```powershell
git add .
git commit -m "Build the container on GitHub Actions"
git push
```

On GitHub, open the repo's **Actions** tab and wait for "test and build container" to go green
(about 5 minutes). Then make the image public, so Azure can pull it without a password:
your GitHub profile > **Packages** > **what-else-was-on** > **Package settings** >
**Change visibility** > **Public**.

## 4. Create the app, with the Foundry model as a secret

This reads the key from your `.env`, so you do not have to paste it:

```powershell
$KEY = (Select-String -Path .env -Pattern '^AZURE_OPENAI_API_KEY=(.*)$').Matches[0].Groups[1].Value.Trim()
az containerapp env list --resource-group $RG -o table
az containerapp create --name whatelse --resource-group $RG --environment whatelse-env `
  --image ghcr.io/raeesalorgat/what-else-was-on:latest --target-port 8000 --ingress external `
  --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 `
  --secrets foundry-key=$KEY `
  --env-vars AZURE_OPENAI_ENDPOINT=https://whatelse-resource.services.ai.azure.com AZURE_OPENAI_API_KEY=secretref:foundry-key AZURE_OPENAI_MODEL=gpt-4.1-mini AZURE_OPENAI_CHECKER_MODEL=gpt-4.1 WHATELSE_LIVE_RUNS=20
```

If `env list` shows no `whatelse-env`, create it first:
`az containerapp env create --name whatelse-env --resource-group $RG --location $REGION`.

- `WHATELSE_LIVE_RUNS=20`: at most 20 live agent runs a day from the public page (each costs tokens).
  Use `0` to switch live runs off; the saved stories still show.
- `--max-replicas 1`: one copy of the app, so new stories and the one-run-at-a-time rule are shared.
- `--min-replicas 0`: the app sleeps when nobody uses it (cheapest). The first visit after a quiet
  spell takes about 20 to 30 seconds to wake up. For live judging you can set `--min-replicas 1`.

## 5. Check it

```powershell
az containerapp show --name whatelse --resource-group $RG --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<that address>` and `https://<that address>/healthz` (should show `{"ok":true,"matches":[1,2,3]}`).

## Updating after code changes

Push to GitHub, wait for the Actions run to go green, then tell the app to pull the new image
(the suffix just has to be new each time, for example the date and a letter):

```powershell
az containerapp update --name whatelse --resource-group $RG --image ghcr.io/raeesalorgat/what-else-was-on:latest --revision-suffix v20261026a
```

The settings and the secret are kept.

## Logs if something goes wrong

```powershell
az containerapp logs show --name whatelse --resource-group $RG --follow
```

## Cleaning up after the hackathon

```powershell
az group delete --name whatelse-app
```

This deletes only the app and its environment; Foundry is in a different resource group.

## Notes

- Stories written by live runs on the hosted page live inside the running container and are lost
  when it restarts or is updated. Stories you make locally with `scripts/run_pipeline.py` and commit
  are baked into the image on each build.
- Run it locally the same way: `docker build -t whatelse .` then
  `docker run -p 8000:8000 --env-file .env whatelse`.
