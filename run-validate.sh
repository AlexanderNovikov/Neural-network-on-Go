#!/usr/bin/env bash
source ~/.bash_profile
export GOPATH=$( cd $(dirname $0) ; pwd -P )/src;
export GOROOT="$HOME/go1.13.1/"
appDir=$(pwd)
cd src;
$HOME/go1.13.1/bin/go run *.go --appDir "${appDir}" --restore --validate
