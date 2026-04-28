#!/usr/bin/env bash

read -p "Delete environment 'ensemble_rep_hgt'? (y/n)" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
  conda remove --name ensemble_rep_hgt --all
fi