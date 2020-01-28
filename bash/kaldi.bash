# This code is intended to be used in all test cases which require Kaldi.
JENKINS_KALDI_ROOT=/net/vol/jenkins/kaldi/
CURRENT_WORKING_KALDI_ROOT=$JENKINS_KALDI_ROOT/2018-08-08_10-09-05_8e97639b7066b5d8a649827975263b6fc288e6ef
CURRENT_BUILDING_KALDI_ROOT=$JENKINS_KALDI_ROOT/2020-01-10_10-32-09_cbdbedefcdd47e02a685c1dc2d1b128c30bdf6b2

if [[ $USER =~ ^jenkins ]]; then
    if [ $# -eq 0 ]; then
            export KALDI_ROOT=$CURRENT_WORKING_KALDI_ROOT
    else
        if [ $1 == "build" ]; then
            export KALDI_ROOT=$CURRENT_BUILDING_KALDI_ROOT
        else
            echo "Error unknown input" $1
            exit 1
        fi
    fi
fi
