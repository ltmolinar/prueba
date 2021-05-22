import os
import argparse
from typing import no_type_check
import pandas as pd
import numpy as  np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pickle



from model import model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='data',
        help='Path to the training data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for SGD'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD'
    )

    args = parser.parse_args()

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    #preparar datos
    df = pd.read_csv('./data/data.csv', delimiter=';')

    #typo de datos adecuados para codificación de varibales categoricas
    df['descTema'] = df['descTema'].astype('str')
    df['descCanalRadicacion'] = df['descCanalRadicacion'].astype('str')
    df['descSegmentoAfiliado'] = df['descSegmentoAfiliado'].astype('str')
    df['descCicloVida'] = df['descCicloVida'].astype('str')
    df['descOcupacion'] = df['descOcupacion'].astype('str')
    df['descRegional'] = df['descRegional'].astype('str')

    #codificación de variables:
    LE = LabelEncoder()
    df2 = df[['afi_hash64','descTema','descSexo', 'descSegmentoAfiliado', 'edadAfiliado', 
          'EstadoPO', 'EstadoPV', 'EstadoCES', 'ultimoIBC','IndicadorUsaClave', 'idAfiliadoTieneClave', 'TieneEmail',
          'descCicloVida','descOcupacion', 'descRegional' , 'descCanalRadicacion']]

    df2 = df2[df2.descCanalRadicacion.isin(['LINEA DE SERVICIO', 'OFICINA DE SERVICIO', 'OFICINA VIRTUAL'])]

    df2['afi_hash64'] = LE.fit_transform(df2['afi_hash64'])
    df2["descTema"] = LE.fit_transform(df2['descTema'])
    df2["descSexo"] = LE.fit_transform(df2['descSexo'])
    df2["descSegmentoAfiliado"] = LE.fit_transform(df2['descSegmentoAfiliado'])
    df2["EstadoPO"] = LE.fit_transform(df2['EstadoPO'])
    df2["EstadoPV"] = LE.fit_transform(df2['EstadoPV'])
    df2["EstadoCES"] = LE.fit_transform(df2['EstadoCES'])
    df2["descCanalRadicacion"] = LE.fit_transform(df2['descCanalRadicacion'])
    df2["IndicadorUsaClave"] = LE.fit_transform(df2['IndicadorUsaClave'])
    df2["idAfiliadoTieneClave"] = LE.fit_transform(df2['idAfiliadoTieneClave'])
    df2["descCicloVida"] = LE.fit_transform(df2['descCicloVida'])
    df2["TieneEmail"] = LE.fit_transform(df2['TieneEmail'])
    df2["descOcupacion"] = LE.fit_transform(df2['descOcupacion'])
    df2["descRegional"] = LE.fit_transform(df2['descRegional'])

    # Eliminamos los registros de clientes que de acuerdo con el conocimiento del negocio no nos contactan normalmente

    df3 = df2.drop(df.index[ (df['edadAfiliado'] > 90) & (df['edadAfiliado'] < 18)])

    #normalización de varibles:
    scaler = MinMaxScaler()
    scaler.fit(df3.iloc[:,1:15])
    df4= scaler.transform(df3.iloc[:,1:15])
    df4 = pd.DataFrame(df4, columns =  ['descTema','descSexo', 'descSegmentoAfiliado', 'edadAfiliado', 
        'EstadoPO', 'EstadoPV', 'EstadoCES', 'ultimoIBC','IndicadorUsaClave', 'idAfiliadoTieneClave', 
        'TieneEmail', 'descCicloVida','descOcupacion', 'descRegional' ])

    #Separación train y test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df4,df3["descCanalRadicacion"], 
        test_size=0.2, random_state=10, stratify =df3["descCanalRadicacion"] )

    #modelo
    sfs1, knn = model()

    #entrenamiento
    sfs1 = sfs1.fit(X_train, y_train)

    X_train_sfs = sfs1.transform(X_train)
    X_test_sfs = sfs1.transform(X_test)

    clfKnn_sfs = knn.fit(X_train_sfs, y_train)

    score_clfknn_sfs = [f1_score(y_test,clfKnn_sfs.predict(X_test_sfs).astype('int64'), average='weighted'), 
                recall_score(y_test, clfKnn_sfs.predict(X_test_sfs).astype('int64'), average='weighted'), 
               precision_score(y_test, clfKnn_sfs.predict(X_test_sfs).astype('int64'), average='weighted')]

    with open('./outputs/knn_sfs_model.pkl', 'wb') as model_pkl:
        pickle.dump(clfKnn_sfs, model_pkl)


