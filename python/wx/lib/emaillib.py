import sys
import itertools

from email.Parser import Parser
from optparse import OptionParser

def args_from_email(str):
    msg = Parser().parsestr(str)

    def get_body(x):
        # get the text from messages of type 'text/plain'
        if x.get_content_maintype() == 'multipart':
            for y in x.get_payload():
                for z in get_body(y):
                    yield z
        elif x.get_content_type() == 'text/plain':
            yield x.get_payload()

    import pdb; pdb.set_trace()
    ret = list(get_body(msg))
    assert len(ret) == 1
    return ret[0].rstrip().split(' ')


#f = open('/home/kleeman/Desktop/sample_email.txt', 'r')
#ret = args_from_email(f.read())
#f.close()


#
#
#fimg = open('/home/kleeman/Desktop/test.jpg', 'rb')
#img = fimg.read()
#img_str = base64.b64encode(zlib.compress(img,9))
#
#def contents(x):
#    if x.get_content_maintype() == 'multipart':
#        return [contents(y) for y in x.get_payload()]
#    else:
#        if x.get_filename():
#            return x.get_payload(decode=True)
#        else:
#            return x.get_payload()
#
##EXAMPLE OF SENDING PICTURES IN AN EMAIL
## Import smtplib for the actual sending function
#import smtplib
#
## Here are the email package modules we'll need
#from email.mime.image import MIMEImage
#from email.mime.multipart import MIMEMultipart
#
#COMMASPACE = ', '
#
## Create the container (outer) email message.
#msg = MIMEMultipart()
#msg['Subject'] = 'Our family reunion'
## me == the sender's email address
## family = the list of all recipients' email addresses
#msg['From'] = me
#msg['To'] = COMMASPACE.join(family)
#msg.preamble = 'Our family reunion'
#
## Assume we know that the image files are all in PNG format
#for file in pngfiles:
#    # Open the files in binary mode.  Let the MIMEImage class automatically
#    # guess the specific image type.
#    fp = open(file, 'rb')
#    img = MIMEImage(fp.read())
#    fp.close()
#    msg.attach(img)
#
## Send the email via our own SMTP server.
#s = smtplib.SMTP()
#s.sendmail(me, family, msg.as_string())
#s.quit()
