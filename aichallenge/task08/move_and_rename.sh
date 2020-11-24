mv data/train/*/* data/train/
mv data/test/*/* data/test/
echo "하위 폴더의 모든 이미지 파일과 XML 파일을 train/로 옮겼습니다."

rm -r data/train/MASK/
rm -r data/test/MASK/
echo "MASK 폴더를 모두 삭제하였습니다."