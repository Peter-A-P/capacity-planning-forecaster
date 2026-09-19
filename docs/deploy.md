# Putting the dashboard on capacity.peterparker.ca

The dashboard is `dashboard/`: static HTML, CSS and JavaScript against precomputed JSON,
written by `headroom export`. There is no build step, no backend and no dependency to
install, so hosting it is a file copy and one DNS record.

`PLAN.md` specifies Azure Static Web Apps on the free tier. The Intervention Targeting
Engine's `targeting.peterparker.ca` was set up the same way first, and its account of what
went wrong is worth reading before starting:
[`docs/deploy.md` in intervention-targeting-engine](https://github.com/Peter-A-P/intervention-targeting-engine/blob/main/docs/deploy.md).

## Done, 2026-09-18

The app is `capacity-peterparker-ca`, alongside the other sites in the same resource group,
and `capacity.peterparker.ca` resolves to it with a managed certificate. What follows is how
it was done and how to do it again.

**Check the subscription before creating anything.** `az staticwebapp list -o table` has to
show the apps that already exist; if it comes back empty the CLI is signed in to the wrong
tenant, and creating the app in whichever subscription happens to be default is how a
personal project ends up billed to the wrong place.

```powershell
az login                         # pick the tenant that holds the existing apps
az account set --subscription "<the subscription>"
az staticwebapp list -o table    # the other sites should be listed
```

Note the **resource group** they are in; the new one goes beside them.

## Order of work

Steps 1 and 3 can be done before the dashboard exists. Nothing is served until step 4, so an
empty site in between is expected rather than a fault.

## 1. Create the Static Web App

Portal: [Azure Portal](https://portal.azure.com) → Create a resource → **Static Web App**.
Microsoft's walkthrough is
[Quickstart: Building your first static site](https://learn.microsoft.com/en-us/azure/static-web-apps/get-started-portal).

| Setting | Value |
|---|---|
| Resource group | the one the other apps are in |
| Name | `capacity-peterparker-ca` |
| Plan type | **Free** |
| Region | any near you; content is served from a CDN regardless |
| Deployment source | **Other** |

**Choose "Other", not GitHub.** Connecting GitHub commits a workflow and an Azure
credential into this repository, which is about to go public. The deployment in step 4
pushes from this machine instead and puts nothing in the repository. Route A in the
Intervention Targeting Engine's runbook is the GitHub alternative, if you ever want the
live page to track `main`.

Or from the CLI:

```powershell
az staticwebapp create --name capacity-peterparker-ca --resource-group <group> --location <region> --sku Free
```

## 2. A second app, not a second hostname

A Static Web App serves one set of files to every hostname attached to it and does not route
by host, so `capacity.peterparker.ca` cannot be a second hostname on the main site's app.
That is why this is its own app, exactly as `targeting` is.

## 3. The DNS record, at Cloudflare

Azure needs a CNAME from the subdomain to the app's generated hostname. A subdomain takes a
single CNAME; only an apex domain needs the TXT validation dance, and this is a subdomain.

1. Copy the app's URL from its **Overview** page: `https://<generated-name>.azurestaticapps.net`.
2. Go to [the Cloudflare dashboard](https://dash.cloudflare.com), open `peterparker.ca`, then
   **DNS** → **Records** → **Add record**
   ([how to create a record](https://developers.cloudflare.com/dns/manage-dns-records/how-to/create-dns-records/)).

   | Type | Name | Target | Proxy status | TTL |
   |---|---|---|---|---|
   | CNAME | `capacity` | `<generated-name>.azurestaticapps.net` | **DNS only** | Auto |

3. **Set proxy status to DNS only, the grey cloud, not the orange one.** This is the step
   that goes wrong. A proxied record hides the real target behind Cloudflare's addresses, so
   Azure's validation cannot see the CNAME it asked for, and the managed TLS certificate is
   never issued. The Intervention Targeting Engine's deployment records the same thing.
   Cloudflare's reference:
   [Proxy status](https://developers.cloudflare.com/dns/proxy-status/).

4. In the Static Web App: **Settings** → **Custom domains** → **+ Add** → **Custom domain on
   other DNS**. Enter `capacity.peterparker.ca`, choose hostname record type **CNAME**, and
   add. Microsoft's page for this is
   [Set up a custom domain with external providers](https://learn.microsoft.com/en-us/azure/static-web-apps/custom-domain-external).

   From the CLI instead:

   ```powershell
   az staticwebapp hostname set --name capacity-peterparker-ca --resource-group <group> --hostname capacity.peterparker.ca
   ```

5. Wait for validation. Usually minutes, occasionally an hour. Azure issues and renews the
   TLS certificate itself once it passes; there is nothing to buy or install.

## 4. Publish the files

`dashboard/` is in the repository, data included, so this needs no build step:

```powershell
$env:SWA_CLI_DEPLOYMENT_TOKEN = az staticwebapp secrets list --name capacity-peterparker-ca --resource-group <group> --query properties.apiKey -o tsv
npx --yes @azure/static-web-apps-cli deploy ./dashboard --env production
```

The token is read straight into the environment of one shell so it never appears on a command
line or in shell history. **It is enough on its own to publish to that site**: it does not go
in the repository, in `CLAUDE.local.md`, or into a chat. Reference:
[Deploy using the SWA CLI](https://learn.microsoft.com/en-us/azure/static-web-apps/static-web-apps-cli-deploy).

Rerun both lines whenever `headroom export` rewrites the JSON.

## 5. Check it

- Before publishing, look at it locally: `uv run headroom serve`, then
  <http://localhost:8080>. **Not `python -m http.server`.** A plain file server sends none of
  the headers in `dashboard/staticwebapp.config.json`, so it shows a page the content
  security policy would partly refuse; `headroom serve` reads that file and sends what the
  host will send. On the Intervention Targeting Engine a plain file server hid a broken
  chart legend on the live site for two weeks while every local check looked correct.
- `https://capacity.peterparker.ca` serves over HTTPS with no certificate warning.
- The page's numbers match the README's tables. They come from the same checkpoints.
- Open the browser console on the live page and confirm it is empty. A content security
  policy violation is reported there and nowhere else.

## If the free tier changes

GitHub Pages serves a subdirectory only from a branch root, so `dashboard/` has to become the
root of a published branch:

```bash
git subtree push --prefix dashboard origin gh-pages
```

Then enable Pages on `gh-pages` and point the same CNAME at `peter-a-p.github.io`. Two things
are worse: Pages requires the repository to be public before it serves anything, and
`staticwebapp.config.json` is ignored, so the content security policy is lost.
