#!/usr/bin/env bash

read -p "Delete environment 'genomic-data-rep'? (y/n)" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
  conda remove --name genomic-data-rep --all
fi