#!/bin/bash

FILE="code.py"
FUNC=$1

awk "
BEGIN{p=0}

/^def ${FUNC}\\(/{
    p=1
}

p==1{
    print
}

/^def / && \$0 !~ /^def ${FUNC}\\(/ && p==1{
    exit
}
" $FILE | head -n -1 | xclip -selection clipboard

echo "$FUNC copied to clipboard"