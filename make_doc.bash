#!/bin/bash

modules=`./show_modules.bash | tee doc/modules.lst`

echo "Found the following modules:"
echo $modules
cd doc
rm index.rst
TOTAL_LINES=`cat template.rst | wc -l`
BEGIN_LINE=`grep -n -e 'available_modules' template.rst | cut -d : -f 1`
TAIL_LINES=$(($TOTAL_LINES-$BEGIN_LINE))
BEGIN_LINE=$(($BEGIN_LINE-1))
head -n $BEGIN_LINE template.rst >> index.rst
cat modules.lst | sed 's,^,* ,g' >> index.rst
tail -n $TAIL_LINES template.rst >> index.rst
make html
cd ..

ln -s doc/_build/html/index.html documentation.html