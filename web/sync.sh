#!/bin/bash

cd ./web_data/
tar cvfz ../web/data.tgz ./4/ ./5/ ./6/ ./7/ ./8/ ./spot/ ./time.bin
cd ../

rsync -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./web/ ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html

ssh -i ~/Downloads/slocum_server.pem "cd /var/www/ensembleweather/public_html; tar xvfz data.tgz;"
