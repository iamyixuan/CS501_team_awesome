from ftplib import FTP
ftp = FTP('ftp.ncei.noaa.gov')
ftp.login()
ftp.cwd('pub/data/asos-fivemin/')
#ftp.cwd('..')
for year in range(2000,2022):
    print(year)
    ftp.cwd('6401-'+str(year))
    #Files
    files = ftp.nlst()
    print(len(files))
    os.mkdir('/scratch/brown/brittoa/data/'+str(year))
    os.chdir('/scratch/brown/brittoa/data/'+str(year))
    for file in range(0,len(files)):
        name = files[file]
        FileData =open(name,"wb")
        ftp.retrbinary(f"RETR {name}", FileData.write)
    ftp.cwd('..')
    os.chdir('/scratch/brown/brittoa/data')
