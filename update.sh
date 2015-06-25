#!/usr/bin/env bash
git pull
git submodule foreach git pull
make_doc