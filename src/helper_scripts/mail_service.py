import smtplib


if __name__ == '__main__':
    print('Enter email address (eg. yourname@gmail.com):')
    gmail_user = input() 
    print('Enter password OR App password for 2-Step-Verification (https://support.google.com/accounts/answer/185833):')
    gmail_password = input()

    sent_to = [gmail_user]
    subject = "Test"
    body = "This is a test report"
    
    msg = "\r\n".join([
        "From:" + gmail_user,
        "To:" + "; ".join(sent_to),
        "Subject:" + subject,
        "",
        body
        ])

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, sent_to, msg)
        server.close()
        print("Message sent! :-)")
    except Exception as e:
        print(e)
