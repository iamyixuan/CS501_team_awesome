#!/bin/csh


foreach yyyy (2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021)
foreach mm (01 02 03 04 05 06 07 08 09 10 11 12)
#foreach yyyy (2021)
#foreach mm (01 02 03)
#change cat for wheverever you want the output data
cat > /scratch/brown/brittoa/ERA5_data/test${yyyy}${mm}.csv <<EOF
MMDDYY_LST,HHMMSS_LST,STID,DDHHMM_UTC,PTYPE
EOF
#change directory to where downloaded data IS
cd /scratch/brown/brittoa/FiveMinData/${yyyy}
echo ${yyyy}${mm}
grep -wE '5' 64010K*${yyyy}${mm}.dat |cut -f 1- -d " " --output-delimiter ","|cut -c 49-84 |sed -e 's/$/,FZ/'|cut -f 1,2,5,6,7 -d, >> /scratch/brown/brittoa/ERA5_data/test${yyyy}${mm}.csv
#grep -wE 'FZRA|FZDZ|FZRASN|FZRAPL|FZRAPLSN|FZRASNPL|FZDZSN|FZDZPL|FZDZPLSN|FZDZSNPL' 64010K*${yyyy}${mm}.dat |cut -f 1- -d " " --output-delimiter ","|cut -c 49-84 |sed -e 's/$/,FZ/'|cut -f 1,2,5,6,7 -d, >> /scratch/brown/brittoa/test${yyyy}${mm}.csv
#grep -w UP 64010K*${yyyy}${mm}.dat |cut -f 1- -d " " --output-delimiter ","|cut -c 49-84 |sed -e 's/$/,UP/'|cut -f 1,2,5,6,7 -d, >> /scratch/brown/brittoa/test${yyyy}${mm}.csv
#grep -wE 'PL|RAPL|PLRA|PLSN|SNPL|DZPL|PLDZ|DZPLSN|DZSNPL' 64010K*${yyyy}${mm}.dat |cut -f 1- -d " " --output-delimiter ","|cut -c 49-84 |sed -e 's/$/,PL/'|cut -f 1,2,5,6,7 -d, >> /scratch/brown/brittoa/test${yyyy}${mm}.csv
#grep -wE 'SN|SNSH|TSSN|SNRA|RASN|SNDZ|DZSN' 64010K*${yyyy}${mm}.dat |cut -f 1- -d " " --output-delimiter ","|cut -c 49-84 |sed -e 's/$/,SN/'|cut -f 1,2,5,6,7 -d, >> /scratch/brown/brittoa/test${yyyy}${mm}.csv
#grep -wE 'RA|TSRA|DZ|RASH|SHRA|TSSHRA|TSRASH' 64010K*${yyyy}${mm}.dat |cut -f 1- -d " " --output-delimiter ","|cut -c 49-84 |sed -e 's/$/,RA/'|cut -f 1,2,5,6,7 -d, >> /scratch/brown/brittoa/test${yyyy}${mm}.csv

end
end

exit

#modified off code by Dr. Michael Baldwin
