#!/usr/bin/env bash

mkdir -p /usr/share/keyrings && \
wget -qO - https://releases.jfrog.io/artifactory/api/v2/repositories/jfrog-debs/keyPairs/primary/public | gpg --dearmor -o /usr/share/keyrings/jfrog.gpg && \
echo "deb [signed-by=/usr/share/keyrings/jfrog.gpg] https://releases.jfrog.io/artifactory/jfrog-debs focal contrib" | tee /etc/apt/sources.list.d/jfrog.list && \
apt-get update && \
apt-get install -y jfrog-cli-v2
