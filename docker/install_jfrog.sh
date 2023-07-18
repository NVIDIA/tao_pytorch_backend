#!/usr/bin/env bash

wget -qO - https://releases.jfrog.io/artifactory/api/gpg/key/public | apt-key add -;
echo "deb https://releases.jfrog.io/artifactory/jfrog-debs xenial contrib" | tee -a /etc/apt/sources.list;
apt-get update;
apt-get install -y jfrog-cli;
