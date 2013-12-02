#!/bin/bash

# AMI ubuntu-precise-12.04-amd64-server-20131003 (ami-6aad335a)
sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install emacs23 apache2 apache2-doc apache2-utils -y
sudo apt-get install libapache2-mod-python -y
sudo apt-get install python-mysqldb -y
sudo apt-get install libapache2-mod-php5 php5 php-pear php5-xcache -y
sudo apt-get install php5-suhosin -y
sudo apt-get install php5-mysql -y

#TODO : add ensembleweather apache config
# /etc/apache2/sites-available/ensemble-weather
sudo a2dissite default
sudo a2ensite ensemble-weather

sudo mkdir /var/www/ensembleweather/
sudo mkdir /var/www/ensembleweather/public_html/
sudo mkdir /var/www/ensembleweather/logs/

sudo apt-get install postfix -y
# https://www.digitalocean.com/community/articles/how-to-install-and-setup-postfix-on-ubuntu-12-04


