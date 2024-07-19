#!/bin/bash
# Check for presence of FDRecoNumuPredFriend tree in the the output root files
# Check that FDRecoNumuPredFriend has the same number of entries as cafTree
# Requires duneutil and root

DIR="/pnfs/dune/persistent/users/awilkins/ND_CAF_fdrecopreds/chasnip-NDCAF_OnAxisHadd-FHC_fdrecopreds"
MISSING_OUT_FILE="/exp/dune/app/users/awilkins/ndfd_translator/grid/missing_friendtree.txt"
INCOMPLETE_OUT_FILE="/exp/dune/app/users/awilkins/ndfd_translator/grid/incomplete_friendtree.txt"

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
    echo $file >> $MISSING_OUT_FILE
  else
    complete_friend=$( \
      root -b \
           -l \
           -q \
           `pnfs2xrootd $file` \
           -e 'std::cout << "Result " << (int)(cafTree->GetEntries() == FDRecoNumuPredFriend->GetEntries()) << std::endl;' 2>/dev/null \
           | grep "Result " \
           | sed "s/Result //" \
    )
    if [[ $complete_friend == 0 ]]
    then
      echo $file >> $INCOMPLETE_OUT_FILE
    fi
  fi
done
