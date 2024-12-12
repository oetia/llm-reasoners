# Development notes

## Developer Setup
### 1. Set up submodule
```
cd examples/
git submodule add git@github.com:BlankCheng/AgentLab.git AgentLab
pip install -e .
```
Refer to [README.md](./README.md) for more steps to set up the AgentLab.
### 2. Install browsergym
```
git submodule add https://github.com/ServiceNow/BrowserGym.git BrowserGym
cd BrowserGym
make install
```

### 3. Set up datasets
Refer to [README.md](./README.md) and [agentlab unofficial readme](https://github.com/gasse/webarena-setup/tree/main/webarena)
For example
**Webarena** - locally hosted

Note: There seems to be a bug in browsergym/playwright which makes using demo_mode on webarena infinitely stall the browser for fill() actions. Be sure to set demo_mode to off in the HighlevelActionSet.

```bash
# download nltk punkt_tab
python -c "import nltk; nltk.download('punkt_tab')"

# set up webarena environment variables
export BASE_URL="http://localhost"
# webarena environment variables (change ports as needed)
export WA_SHOPPING="$BASE_URL:8082/"
export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
export WA_REDDIT="$BASE_URL:8080"
export WA_GITLAB="$BASE_URL:9001"
export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:443"
export WA_HOMEPAGE="$BASE_URL:80"

# if your webarena instance offers the FULL_RESET feature (optional)
export WA_FULL_RESET="$BASE_URL:7565"

# otherwise, be sure to NOT set WA_FULL_RESET, or set it to an empty string
export WA_FULL_RESET=""

# set up openai api key
export OPENAI_API_KEY=...
```

Then serve the webarena server following the [webarena instructions](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md). e.g., for the shopping instance (note the .tar file needs downloading), run
```bash
# 1. shopping
docker load --input shopping_final_0712.tar
docker run --name shopping -p 8082:80 -d shopping_final_0712
# wait ~1 min to wait all services to start

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:8082" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:8082" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush

# 2. shopping_admin
docker load --input shopping_admin_final_0719.tar
docker run --name shopping_admin -p 8083:80 -d shopping_admin_final_0719
# wait ~1 min to wait all services to start
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:8083" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:8083/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
```

Test the shopping instance by running
```bash
curl $WA_SHOPPING
```

**VisualWebArena** -- locally hosted
```bash
# visualwebarena environment variables (change ports as needed)
export VWA_CLASSIFIEDS="$BASE_URL:8083"
export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export VWA_SHOPPING="$BASE_URL:8082"
export VWA_REDDIT="$BASE_URL:8080"
export VWA_WIKIPEDIA="$BASE_URL:8081"
export VWA_HOMEPAGE="$BASE_URL:80"

# if your webarena instances offers the FULL_RESET feature (optional)
export VWA_FULL_RESET="$BASE_URL:7565"
```

Then serve the visualwebarena server following the [visualwebarena instructions](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md). e.g., for the reddit instance (note the .tar file needs downloading), run
```bash
docker load --input postmill-populated-exposed-withimg.tar
docker run --name forum -p 8080:80 -d postmill-populated-exposed-withimg
```


## Visualize AgentXray
```bash
export AGENTXRAY_SHARE_GRADIO=true
agentlab-xray
```