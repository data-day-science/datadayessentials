coverage run -m pytest datascience_core
coverage report --omit="*/test*","*/_base.py"
coverage html --omit="*/test*","*/_base.py" -d ./.reports/coverage/ 
coverage xml --omit="*/test*","*/_base.py" -o ./.reports/coverage/coverage.xml
genbadge coverage -i ./.reports/coverage/coverage.xml
mv ./coverage-badge.svg ./.reports/coverage/coverage-badge.svg
