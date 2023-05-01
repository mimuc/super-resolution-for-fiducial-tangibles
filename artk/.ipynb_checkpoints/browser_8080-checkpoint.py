# import HTMLSession from requests_html
from requests_html import HTMLSession
 
# create an HTML Session object
session = HTMLSession()
 
# Use the object above to connect to needed webpage
resp = session.get("http://127.0.0.1:8080")
 
# Run JavaScript code on webpage
resp.html.render(sleep=600000, keep_page=True)