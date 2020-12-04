"""Implement a logging system.

This class will direct output to both the screen and a specified file.

Code for the article "Monte Carlo Methods for the Neutron Transport Equation.

By A. Cox, A. Kyprianou, S. Harris, M. Wang.

Thi sfile contains the code to produce the plots in the case of the 2D version
of the NTE.

MIT License

Copyright (c) Alexander Cox, 2020.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
from pathlib import Path


class Logger(object):
    """Send output to file and screen.

    Based on an answer here:
        https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """

    def __init__(self):
        self.terminal = sys.stdout
        log_dir = os.getcwd()+'/output'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir+"/logfile.log", "a")

    def write(self, message):
        """Send message to screen and file."""
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        """Close open filestream. Revert to screen only output."""
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        """Not sure if needed."""
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
