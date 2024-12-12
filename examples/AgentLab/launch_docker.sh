# Launch shopping
docker run --name shopping -p 8082:80 -d shopping_final_0712
sleep 15
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:8082" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:8082" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush

# Launch shopping_admin
docker run --name shopping_admin -p 8083:80 -d shopping_admin_final_0719
sleep 15 
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:8083" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:8083/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flu

# Launch forum
docker run --name forum -p 8080:80 -d postmill-populated-exposed-withimg

