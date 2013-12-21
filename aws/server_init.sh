#!/bin/bash

# TODO add ssl and encryption

# fail on any error
set -e
# this file should be in slocum/aws
sudo apt-get install -y realpath
DIR=`dirname $0`
DIR=`realpath $DIR`

function install_packages {
  # AMI ubuntu-precise-12.04-amd64-server-20131003 (ami-6aad335a)
  sudo apt-get install emacs23 apache2 apache2-doc apache2-utils -y
  sudo apt-get install libapache2-mod-python realpath -y
  sudo apt-get install python-mysqldb -y
  sudo apt-get install libapache2-mod-php5 php5 php-pear php5-xcache -y
  sudo apt-get install php5-suhosin -y
  sudo apt-get install php5-mysql -y
  sudo apt-get install amavisd-new -y
  sudo apt-get install spamassassin spamc -y
  sudo apt-get install clamav -y
}

function postfix {
  debconf-set-selections "postfix postfix/mailname string your.hostname.com"
  debconf-set-selections "postfix postfix/main_mailer_type string 'Internet Site'"
  sudo apt-get install -y -q --force-yes postfix
  # https://www.digitalocean.com/community/articles/how-to-install-and-setup-postfix-on-ubuntu-12-04
  # took most of this from http://flurdy.com/docs/postfix/#data
  sudo postconf "myorigin=ensembleweather.com"
  sudo postconf "smtpd_helo_required=yes"
  sudo postconf "smtpd_delay_reject=yes"
  sudo postconf "disable_vrfy_command=yes"
  # Requirements for the HELO statement
  sudo postconf "smtpd_helo_restrictions=permit_mynetworks, warn_if_reject reject_non_fqdn_hostname, reject_invalid_hostname, permit"
  # Requirements for the sender details
  sudo postconf "smtpd_sender_restrictions = permit_mynetworks, warn_if_reject reject_non_fqdn_sender, reject_unknown_sender_domain, reject_unauth_pipelining, permit"
  # Requirements for the connecting server 
  sudo postconf "smtpd_client_restrictions = reject_rbl_client sbl.spamhaus.org, reject_rbl_client blackholes.easynet.nl"
  # Requirement for the recipient address
  sudo postconf "smtpd_recipient_restrictions = reject_unauth_pipelining, permit_mynetworks, reject_non_fqdn_recipient, reject_unknown_recipient_domain, reject_unauth_destination, permit"
  sudo postconf "smtpd_data_restrictions = reject_unauth_pipelining"
  # not sure of the difference of the next two
  # but they are needed for local aliasing
  sudo postconf "alias_maps = hash:/etc/aliases"
  sudo postconf "alias_database = hash:/etc/aliases"

  # anti-virus
  sudo postconf "content_filter = amavis:[127.0.0.1]:10024"
  sudo postconf "allow_mail_to_commands = alias,forward,include"
  # anti-spam
  sudo sed -i 's/ENABLED=0/ENABLED=1/g' /etc/default/spamassassin
  # anti-virus

  sudo cp $DIR/aliases /etc/aliases
  sudo cp $DIR/master.cf /etc/postfix/master.cf
  sudo postalias /etc/aliases
  sudo postfix reload

  # make a .forward file
  echo 'ubuntu, "|/home/ubuntu/test.py"' > /home/ubuntu/.forward
}

function website {
  # Link to the website
  sudo mkdir -p /var/www/ensembleweather/logs
  sudo ln -s $DIR/public_html /var/www/ensembleweather/public_html
  sudo ln -s $DIR/ensembleweather.a2cfg /etc/apache2/sites-available/ensemble-weather
  sudo a2dissite default
  sudo a2ensite ensemble-weather
  sudo service apache2 reload
}

function slocum {
  sudo apt-get install python-pip python-dev  libhdf5-openmpi-dev netcdf-bin libnetcdf-dev -y
  sudo apt-get install libatlas-dev liblapack-dev libblas-dev build-essential gfortran -y
  sudo pip install numpy
  sudo pip install BeautifulSoup
  sudo pip install netCDF4
  sudo apt-get install python-scipy python-matplotlib -y
  sudo pip install iris
  cd ~/
  git clone https://github.com/akleeman/scidata.git
  echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/scidata/src/:/home/ubuntu/slocum/python" >> ~/.bashrc
}

function swap {
  sudo dd if=/dev/xvda1 of=/swapfile bs=1024 count=512k
  sudo mkswap /swapfile
  sudo swapon /swapfile
  sudo su -
  sudo echo "/swapfile       none    swap    sw      0       0 " >> /etc/fstab
  sudo echo 0 > /proc/sys/vm/swappiness
  exit
}

install_packages
postfix
website

