#!/bin/bash

rsync -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./map.html ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html/index.html
rsync -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./slocum.js ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html/
rsync -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./js/TimeSlider.js ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html/js/
rsync -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./data/time.bin ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html/data/time.bin

for f in 4 5 6 7 8 spot; do
  rsync -avzhe "ssh -i ~/Downloads/slocum_server.pem" ./data/$f/ ubuntu@52.5.75.36:/ebs/var/www/ensembleweather/public_html/data/$f
done

