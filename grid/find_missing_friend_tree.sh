#!/bin/bash
# Check for presence of FDRecoNumuPredFriend tree in the the output root files
# Requires duneutil and root

DIR="/pnfs/dune/persistent/users/awilkins/ND_CAF_fdrecopreds/chasnip-NDCAF_OnAxisHadd-FHC"
OUT_FILE="/exp/dune/app/users/awilkins/ndfd_translator/missing_friendtree.txt"

cd $DIR

for file in *
do
  echo $file
  has_friend=$( \
    root -b \
         -l \
         -q \
         `pnfs2xrootd $file` \
         -e 'std::cout << "Result " << _file0->GetListOfKeys()->Contains("FDRecoNumuPredFriend") << std::endl;' 2>/dev/null \
         | grep "Result " \
         | sed "s/Result //" \
  )
  if [[ $has_friend == 0 ]]
  then
    echo $file >> $OUT_FILE
  fi
done

