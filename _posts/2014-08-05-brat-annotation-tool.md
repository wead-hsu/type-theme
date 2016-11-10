---
layout: default
title: An Annotation Tool - Brat
categories: common-utils
---

### 1. What is brat?

- Brat is a web-browser based nlp annotation tool by which members in nlp group can annote data collaboratively.




### 2. How to install brat?

+ At first, you need a web server, for example apache
	1. sudo apt-get install apache2
	2. if you want to use directory in your home, i.e. /home/'usrname'/public_html

```
cp /etct/apache2/sites-available/default /etc/apache2/sites-available/usr-public [whatever you like] 
change the directory in 'usr-public' to /home/'usrname'/public_html
sudo a2dissite default && a2ensite usr-public
sudo /etc/init.d/apache2 reload/restart
```

+ and then you need to configure the CGI property for apache2. To do so, you need to

	1 configure httpd.conf in  /etc/apache2/(see blow) and change the CGI directory in /etc/apache2/sites-available/usr-public to /home/'usrname'/public_html [not the sub directory in public_html]. 

```
<Directory /home/*/public_html>
AllowOverride Options Indexes FileInfo Limit
AddType application/xhtml+xml .xhtml
AddType font/ttf .ttf
# For CGI support
AddHandler cgi-script .cgi
# Comment out the line above and uncomment the line below for FastCGI
#AddHandler fastcgi-script fcgi
</Directory>

\# For FastCGI, Single user installs should be fine with anything over 8
\#FastCgiConfig -maxProcesses 16
```

+ Install brat:

	* simply unzip the file download from offical website and copy it to the web server directory, say /home/'usrname'/public_html here, and 

　　　　　　sudo sh ./install.sh

+ Test:

	* open your browser and go to url 'http://localhost/brat/index.xhtml'. If every thing works well, there will be an interface with 'welcome', otherwise error messages will be delivered. Typically, there will be 3 types of errors like: actiongetcollectioninformation .... .... ....　

	* if you encounter with this situation, you may refer to debug tool troubleshooting.sh in drat/tools and look into error_log of apache for more information [the default error_log location is configured in /etc/apache2/envvras ]. Make sure that your files have appropriate permissions. 

### 3. How to use it?　

* Follow the procedure:
	* log in.
	* put your files in data folder.
	* generate .ann files by
	* find data -name '*.txt' \| sed -e 's\|\.txt\|.ann\|g' \| xargs touch
	* chmod files
	* build your own annotation.conf file. It should have the available ner tags in this file if you are to assign ner data.
