'''function will take in a matplotlib figure and create an svg, pdf and pdf_tex file
There is probably a few bugs:

HOW TO USE

import line:
from savepdf_tex import savepdf_tex

see func doc sting for how to call the function

Other useful info:

-  For labels which have stuff you want to be interpretered by latex... example
plt.xlabel(r'\$ \frac{h}{H} \$') the r' means its a raw string and the \ before the $ escapes the $.  then just write your latex line normally.
There have been times where you've needed {{}} where in latex only {} is needed.

- If using plt.pcolor or any sort of heavily colored plot (plt.contourf), try and rasterize in python first!
e.g. plt.pcolor(..., rasterized=True)'''


import matplotlib as mpl
import os
import subprocess
from PyPDF2 import PdfFileReader
import re
import shutil
#
def savepdf_tex(fig, fig_loc, name):
    """function will take in a matplotlib figure and create an svg, pdf and pdf_tex file

    Arguments:
        fig {matplotlib.pyplot.figure} -- this is the matplotlib figure to save -
                                            e.g. initialise your matplotlib figure
                                            with fig=plt.figure() then pass fig in
                                            here.
        fig_loc {str} -- FULL path of where you want to save the figure
        name {str} -- File name (no file extension)
    """

    mpl.use('svg')
    new_rc_params = {
        "font.family": 'Times',
        "font.size": 12, #choosing the font size helps latex to place all the labels, ticks etc. in the right place
        "font.serif": [],
        "svg.fonttype": 'none'} #to store text as text, not as path
    mpl.rcParams.update(new_rc_params)
    fig.savefig(f'{fig_loc}{name}.svg', dpi = 1000, format = 'svg', bbox_inches = 'tight') # depends on your final figure size, 1000 dpi should be definitely enough for A4 documents
    incmd = ["inkscape", f'{fig_loc}{name}.svg', "--export-pdf={}.pdf".format(f'{fig_loc}{name}'),
             "--export-latex" ]#,"--export-ignore-filters"]
    subprocess.check_output(incmd)


    # read number of pages in pdf file
    pdf = PdfFileReader(open(f'{fig_loc}{name}.pdf','rb'))
    n_pages_pdf = pdf.getNumPages()
    print(f'# pages in pdf: {n_pages_pdf}')

    # find number of pages in pdf_tex file - use regular expresions
    tex_name = f'{fig_loc}{name}.pdf_tex'
    pattern = re.compile('page=\d+')
    n_pages_tex = 0
    with open(tex_name, 'r') as f:
        for line in f:
            res = re.search(pattern, line)
            if res:
                i = int(res.group().replace('page=', ''))
                n_pages_tex = max(i, n_pages_tex)
                # save line form
                page_line = line

    print(f'# pages in pdf_tex: {n_pages_tex}')

    # make new, corrected pdf_tex file if it is needed
    tex_name_new = tex_name + '_new'

    if n_pages_tex < n_pages_pdf:
        # in this case you need to add/include additional pages
        with open(tex_name, 'r') as f_old:
            with open(tex_name_new, 'w') as f_new:
                for line in f_old:
                    if r'\begingroup%' in line:
                        # define the the text size on the figure
                        f_new.write(r'\begingroup\footnotesize%' '\n')
                        continue

                    if '\end{picture}%' not in line:
                        f_new.write(line)
                    else:
                        # add missing pages
                        for i in range(n_pages_tex+1, n_pages_pdf+1):
                            # page_line - use saved form of line
                            res = re.search(pattern, page_line)
                            old_part = res.group()
                            new_part = 'page=%d' % i
                            f_new.write(page_line.replace(old_part, new_part))
                        f_new.write(line)


    elif n_pages_tex > n_pages_pdf:
        # you need to delete included pages that not exist in pdf file
        with open(tex_name, 'r') as f_old:
            with open(tex_name_new, 'w') as f_new:
                for line in f_old:
                    if r'\begingroup%' in line:
                        # define the the text size on the figure
                        f_new.write(r'\begingroup\footnotesize%' '\n')
                        continue

                    res = re.search(pattern, line)
                    if res:
                        # if 'page=' is in line, check the numeber
                        i = int(res.group().replace('page=', ''))
                        if i <= n_pages_pdf:
                            f_new.write(line)
                        else:
                            continue
                        # you have a problem here, don't rewrite line to new file
                    else:
                        # rewrite all lines without 'page='
                        f_new.write(line)
        shutil.move(tex_name_new, tex_name)

    elif n_pages_tex == n_pages_pdf:
        with open(tex_name, 'r') as f_old:
            with open(tex_name_new, 'w') as f_new:
                for line in f_old:
                    if r'\begingroup%' in line:
                        # define the the text size on the figure
                        f_new.write(r'\begingroup\footnotesize%' '\n')
                    else:
                        f_new.write(line)
        shutil.move(tex_name_new, tex_name)
    mpl.use('Qt5Agg')