# laya-service

Thin FastAPI wrapper around the `laya` decision model, meant to run as its
own Railway service and be called by Paperclip agents for fast, cheap,
non-generative routing/classification decisions.

## Deploy

1. Push this folder to a new GitHub repo, e.g. `yourname/laya-service`:

   ```
   cd laya-service
   git init
   git add .
   git commit -m "Laya decision service"
   gh repo create yourname/laya-service --public --source=. --push
   ```

   (No `gh` CLI? Create the repo on github.com first, then:
   `git remote add origin https://github.com/yourname/laya-service.git`
   `git push -u origin main`)

2. Tell Claude the repo name (`yourname/laya-service`) and it can deploy it
   as a new Railway service in the same project as Paperclip via
   `create-deployment`.

3. Optionally set `API_KEY` as an env var on the Railway service, then send
   `Authorization: Bearer <API_KEY>` from Paperclip when calling it. Without
   this, anyone with the URL can call your endpoint.

## API

`POST /predict`

```json
{
  "state": {"body": "I was charged twice for invoice #4411"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs and outages",
        "sales": "pricing"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the sender ask for money back?"
    }
  }
}
```

Returns `{"result": {...typed answers with probabilities...}}`.

`GET /health` — liveness/readiness check.
