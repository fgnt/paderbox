# This code is intended to be used in all test cases which require Kaldi.
JENKINS_KALDI_ROOT=/net/vol/jenkins/kaldi/
CURRENT_WORKING_KALDI_ROOT=$JENKINS_KALDI_ROOT/2018-08-08_10-09-05_8e97639b7066b5d8a649827975263b6fc288e6ef
CURRENT_BUILDING_KALDI_ROOT=$JENKINS_KALDI_ROOT/2020-09-23_09-43-55_5e6f08c13efc8ab425b77debcb33bae13dc6b31e

if [[ $(hostname) =~ ntjenkins ]]; then
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
