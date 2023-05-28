sudo flask run --cert=/etc/letsencrypt/live/dalleapi.com/fullchain.pem \
  --key=/etc/letsencrypt/live/dalleapi.com/privkey.pem \
  --host 0.0.0.0 \
  --port 443
