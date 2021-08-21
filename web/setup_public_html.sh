#!/bin/bash -ex

cd public_html/

# Move current data to old data
cp -r ./new_data/ ./old_data
rm ./data
ln -s ./old_data ./data

# Overwrite new_data
rm -rf ./new_data
mkdir new_data
cd new_data
tar xvfz ../data.tgz
cd ..

# Point to new data
rm ./data
ln -s ./new_data ./data
rm -rf ./old_data

chown ubuntu:www-data -R ./
chmod u=rwX,g=srX,o=rX -R ./
