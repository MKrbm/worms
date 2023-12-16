#!/bin/bash

# n: Unlink the symbolic links under link

echo "Removing symbolic links under link"

find link -type l -exec unlink {} \;
