# Reject Option

引継ぎ用コードです．
RejectOption-master_paperは修士論文で使用したコードで，プロット用のコードやバッチファイルが入ってますが，引継ぎでは排除しています．
また，閾値ベースのコードが格段に早くなっているので，こちらを使うことを推奨します．
## threshold based rejection
main_thresholdを使用してください．
コマンドライン引数は以下の通り．
- dataset : データセット名，例) iris
- algorithmID : results直下のディレクトリ名
- experimentID : 何試行目の結果かを管理する．※「trial」の後に試行番号を2桁で指定する必要あり．例) trial00，algorithmID直下のディレクトリ名となる．
- fname_train : 学習用データのファイル名，相対パスで指定する．例) ../dataset/iris/a0_0_iris-10tra.dat
- fname_test : 評価用データのファイル名，相対パスで指定する．例) ../dataset/iris/a0_0_iris-10tst.dat
- clf_name : 使用するファジィ識別器の情報があるファイル名．MoFGBMLの場合，VAR-最終個体評価.csvを指定する．

※時間がなかったので修正できてないが，pipelineを使用しているが，機能を全く使ってないので，時間があれば削除してください．

## scikit-learn
機械学習モデルの追加やグリッドサーチのパラメータを変更したい場合は，GridSearchParameter.py内を参照してください．
グリッドサーチにはscikit-learnのGridSearchCVを使用しています．

### コマンドラインでのモデルの指定
コマンドラインでモデルを指定する場合は下記のように入力してください．
- AdaBoost : "Adaboost", 
- Decision Tree : "DecisionTree",
- Naive Bayes : "NaiveBayes",
- Gaussian Process : "GaussianProcess",
- K-Nearest Neighbor : "kNN",
- Multi-Layer Perceptron : "MLP",
- Random Forest : "RF",
- Linear SVC (線形SVM) : "LinerSVC"
- RBFSVC (RBFカーネルSVM) : "RBFSVC"

## Rejector based rejection
### Eric's method
EricのWACの論文を実装．\
V. E. Michael, N. Masuyama and Y. Nojima, “Error-reject tradeoff analysis on two-stage classifier design with a reject option,” In Proc. of 2022 World Automation Congress (WAC), pp. 312-317, Oct. 2022.\\

Ericの論文の実験の場合，main_Ericを使用してください．
コマンドライン引数は以下の通り．
- dataset : データセット名，例) iris
- base_model : results直下のディレクトリ名，ベースとなるモデルを指定する．
- experimentID : 何試行目の結果かを管理する．※「trial」の後に試行番号を2桁で指定する必要あり．例) trial00，algorithmID直下のディレクトリ名となる．
- fname_train : 学習用データのファイル名，相対パスで指定する．例) ../dataset/iris/a0_0_iris-10tra.dat
- fname_test : 評価用データのファイル名，相対パスで指定する．例) ../dataset/iris/a0_0_iris-10tst.dat

### 川野修士論文
ベースモデルをファジィ識別器にしたEric's method
main_fuzzy_rejector.pyを使用してください．
コマンドライン引数は以下の通り．
- dataset : データセット名，例) iris
- algorithmID : results直下のディレクトリ名
- experimentID : 何試行目の結果かを管理する．※「trial」の後に試行番号を2桁で指定する必要あり．例) trial00，algorithmID直下のディレクトリ名となる．
- fname_train : 学習用データのファイル名，相対パスで指定する．例) ../dataset/iris/a0_0_iris-10tra.dat
- fname_test : 評価用データのファイル名，相対パスで指定する．例) ../dataset/iris/a0_0_iris-10tst.dat
- clf_name : 使用するファジィ識別器の情報があるファイル名．

## Ensemble Rejector
Ericの手法では学習用パターンを１つのモデルにより判定しているが，本手法は複数のモデルの多数決により，棄却するかを判定する．
また，機械学習モデルにより棄却されるパターンを除いた学習用データを使用し，ファジィ識別器を学習することで，ルール数やルール長などを単純化できるのではないかと予測している．

構成としては以下の順となる．
- 学習用データからアンサンブル棄却モデルにより，棄却しないパターンのみ抽出．
- 抽出されたデータを学習用データとして使用し，MoFGBMLを実行し，ファジィ識別器群を獲得
- ファジィ識別器群の内，学習用データの識別精度最大の識別器を用いて，アンサンブル棄却モデルを適用．
- 元の学習用データで学習したファジィ識別器群と，抽出した学習用データで学習したファジィ識別器群との識別精度とルール数，ルール長の比較を行う．また，抽出したデータの分布を見る必要がある．さらに，アンサンブル棄却モデルを適用した際の，識別精度と棄却率を比較する．

プログラムを実行する際には，以下の順に実験を進める．
- main_make_ensemble_dataset.pyを実行
- 得られたデータセットを学習用データとして指定して，MoFGBMLを実行
- main_ensemble_rejector.pyを実行

また，元の学習用データと比較するため，以下を実行する必要がある．
- MoFGBMLを元の学習用データを指定して実行．
- main_ensemble_rejector.pyを実行


