#!/usr/bin/env bash

# first of all wget the dlc_weights from Google Drive
export fileid=1SQHfYgvueH7cgT_XCN0dQX22AY6L4qkT
export filename=dlc_weights.zip
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

unzip -o dlc_weights.zip
rm dlc_weights.zip
