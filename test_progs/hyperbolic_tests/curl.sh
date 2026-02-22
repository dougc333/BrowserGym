curl -sS -D /tmp/hdrs.txt \
  -H "Authorization: Bearer $HYPERBOLIC_API_KEY" \
  -H "Content-Type: application/json" \
  https://api.hyperbolic.xyz/v1/models | head -c 2000

echo
echo "---- response headers ----"
sed -n '1,40p' /tmp/hdrs.txt
