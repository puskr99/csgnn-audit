#!/usr/bin/env bash
# Proves that every file of the original CSGNN release is unchanged in the current checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
ORIG=aef3136bd96b88ebee07cafc209fa9105a58a555
git cat-file -e "$ORIG^{commit}" || { echo "commit $ORIG not found"; exit 1; }
echo "original commit: $(git show -s --format='%H  %an  %ad  %s' $ORIG)"
# every path tracked at the original commit must be identical now (added files live only in audit/, reeval/ and .github/)
if git diff --quiet "$ORIG" HEAD -- $(git ls-tree -r --name-only "$ORIG"); then
  echo "OK: all $(git ls-tree -r --name-only $ORIG | wc -l) original files are byte-identical to $ORIG"
else
  echo "MISMATCH:"; git diff --stat "$ORIG" HEAD -- $(git ls-tree -r --name-only "$ORIG"); exit 1
fi
echo "files added on top of the original release:"
git diff --name-only "$ORIG" HEAD | sed 's/^/  + /'
