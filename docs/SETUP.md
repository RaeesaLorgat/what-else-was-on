# Setup: Azure, Foundry and this repo

Goal: a model deployed in Microsoft Foundry answers a test prompt and calls a tool.
About 45 to 60 minutes. Do the steps in order.

## 1. Azure for Students (about 10 min)

1. Go to https://azure.microsoft.com/free/students and sign up. The student check needs your **Wits student email**. This is separate from your hackathon registration, so it does not matter that you registered with your personal email.
2. You get USD 100 of credit for 12 months, with no credit card needed.
3. When it finishes, open https://portal.azure.com and check you see a subscription called **Azure for Students**.

If the student sign-up is refused, tell me before trying anything else. A pay-as-you-go account needs a card, so don't create one without deciding first.

## 2. Create a Foundry project and deploy a model (about 15 min)

1. Go to https://ai.azure.com and sign in with the same account.
2. Create a new **project**. When asked for a resource:
   - Subscription: Azure for Students
   - Resource group: create one called `rg-whatelse`
   - Region: pick one close by that offers the model (try **Sweden Central** or **East US 2** if the first choice has no quota)
3. In the project, open **Models + endpoints** (or the model catalogue) and deploy a small, cheap chat model. Good choices: `gpt-4.1-mini` or `gpt-4o-mini`.
   - Deployment type: Global Standard is fine.
   - **Write down the deployment name** exactly as shown (often the same as the model name).
4. Open the deployment's page and copy:
   - the **endpoint**: the part that looks like `https://<your-resource-name>.openai.azure.com`
   - one of the **keys**

Cost check: a mini model costs a fraction of a cent per call, so the smoke test costs almost nothing. The USD 100 credit covers the whole hackathon if we stick to mini models.

## 3. Get the repo running (about 10 min)

Open a terminal in the `Premier League Hackathon` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and replace the three placeholder values with your endpoint, key and deployment name.

Run the tests (no Azure needed):

```bash
python -m pytest -q
```

Expected: `2 passed`.

Then the real check:

```bash
python scripts/smoke_test.py
```

Expected output (wording of the answers will differ):

```
Calling deployment 'gpt-4.1-mini' at https://<your-resource-name>.openai.azure.com/openai/v1/
Check 1, model reply: OK, connected.
Check 2, tool called: distance_m({'x1': 40, 'y1': 30, 'x2': 58, 'y2': 22}) -> 19.7
Check 2, final answer: The teammate is about 19.7 m away.
All checks passed.
```

Common errors:

| Error | Fix |
|---|---|
| `Missing or placeholder values in .env` | `.env` still has `<...>` placeholders, or is named `.env.txt` |
| `401` | Wrong key, or key from a different resource than the endpoint |
| `404` / `DeploymentNotFound` | `AZURE_OPENAI_MODEL` must be the **deployment** name, not the model family |
| `429` | Quota hit; wait a minute or deploy in another region |

## 4. GitHub (about 10 min)

1. In the terminal, make the first commit. The two `git config` lines set your **personal** name and email for this repo only; your other repos keep your student identity:

```bash
git init -b main
git config user.name "Raeesa Lorgat"
git config user.email "<the email on your personal GitHub account>"
git config user.email   # check it prints your personal email
git add .
git status        # check .env and .venv are NOT listed
git commit -m "Project setup and Foundry connection check"
```

2. Signed in to your **personal** GitHub account, create a **new public repository** called `what-else-was-on` with **no** README or licence (this folder already has them).
3. Connect and push. Put your personal username in the URL **twice** (before the `@` and in the path). The one before the `@` tells Windows to use your personal GitHub login instead of the saved student one; a browser window asks you to sign in the first time:

```bash
git remote add origin https://<personal-username>@github.com/<personal-username>/what-else-was-on.git
git push -u origin main
```

If the push says `Permission denied` or `403`, Windows used your student login anyway: open **Credential Manager → Windows Credentials**, remove the entry `git:https://github.com`, and push again.

On GitHub, open the commit and check it shows your personal account as the author.

Check on GitHub that `.env` is **not** in the repo. Only `.env.example` should be there.

## 5. Innovation Studio

When the email arrives, set up your profile and add the repo link.
