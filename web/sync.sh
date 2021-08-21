#!/bin/bash -x

echo "Creating Tarball"
cd data;
tar cvfz ../data.tgz ./4/ ./5/ ./6/ ./7/ ./8/ ./spot/ ./time.bin > sync.log
cd ../;

rsync --no-links -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./public_html/ ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html
scp -i ~/Downloads/slocum_server.pem ./data.tgz ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html/data.tgz
scp -i ~/Downloads/slocum_server.pem ./setup_public_html.sh ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/

echo "Exploding Tarball"
ssh -i ~/Downloads/slocum_server.pem ubuntu@52.5.75.36 "cd /var/www/ensembleweather/; ./setup_public_html.sh" >> sync.log
