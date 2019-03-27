# This code is intended to be used in all test cases which require Kaldi.
# This file can be sourced in your own .bashrc or .env file.
if [[ $(hostname) =~ ^ntsim.*|ntpc9|ntjenkins ]];
then
    export KALDI_ROOT=/net/vol/jenkins/kaldi/2018-08-08_10-09-05_8e97639b7066b5d8a649827975263b6fc288e6ef
fi