import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,  cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sayfa yapılandırması
st.set_page_config(
    page_title="Proje Ödevi",
    page_icon=":guardsman:",
    layout="centered"
)

# Tema ve stil özelleştirmeleri
st.markdown(
    """
    <style>
    body {
        background-color: #e6f0ff !important;  /* Açık Mavi Arka Plan */
    }
    .sidebar .sidebar-content {
        background-color: #f7f7f7 !important;  /* Gri Sidebar */
    }
    h1, h2, h3 {
        color: #FF7F7F ;  /* Açık kırmızı Başlıklar */
    }
    .stButton > button {
        background-color: #8B0000;  /* Koyu Kırmızı Butonlar */
        color: white;
        font-size: 18px;  /* Buton yazı boyutunu büyütme */
        padding: 15px 30px;  /* Butonun içindeki boşlukları artırma */
        border-radius: 8px;  /* Buton köşe yuvarlama */
        width: 100%;  /* Butonun genişliğini artırma */
        margin: 10px 0;  /* Butonlar arasına boşluk bırakma */
    }
    .stButton > button:hover {
        background-color: #ff4d4d;  /* Buton Hover Efekti (daha açık kırmızı) */
    }
    .stTextInput input {
        background-color: white;
        color: #333333;  /* Koyu gri metin */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Başlık
st.title("Makine Öğrenmesi Projesi")
st.write("Yapmak istediğiniz gerekli işlem için butonlara basabilirsiniz.")


def perform_ml():
    try:
        # Veri setini yükle
        df = pd.read_csv("diabetes.csv")
        st.subheader("Ham Veriler")
        st.write(df)

        model_name = st.selectbox("Bir Model Seçin", ["Logistic Regression", "SVM", "KNN"])

        # Veriyi hazırlama
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model tanımlama ve eğitim
        model = None
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "SVM":
            model = SVC(kernel='linear', random_state=42)  # Linear SVM
        elif model_name == "KNN":
            model = KNeighborsClassifier()

        if model:
            # Modeli eğitme
            model.fit(X_train, y_train)
            st.write(f"{model_name} model başarıyla eğitildi.")

            # Model tahminleri
            y_pred = model.predict(X_test)

            # Performans metrikleri
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)  # Duyarlılık
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Özgüllük
            f1 = f1_score(y_test, y_pred, zero_division=1)

            # Sonuçları gösterme
            st.subheader(f"{model_name} Model Sonuçları")

            # Karışıklık Matrisi Görselleştirme
            st.write("### Karışıklık Matrisi (Görsel)")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"], ax=ax)
            ax.set_xlabel("Tahmin Edilen")
            ax.set_ylabel("Gerçek")
            ax.set_title(f"{model_name} Karışıklık Matrisi")
            st.pyplot(fig)

            # Performans Metrikleri
            st.write("### Performans Metrikleri")
            st.write(f"- **Doğruluk (Accuracy):** {accuracy:.2f}")
            st.write(f"- **Duyarlılık (Recall):** {recall:.2f}")
            st.write(f"- **Özgüllük (Specificity):** {specificity:.2f}")
            st.write(f"- **F1-Skor:** {f1:.2f}")

        else:
            st.warning("Lütfen bir model seçin!")

    except KeyError as e:
        st.error(f"Beklenmeyen bir hata oluştu: {e}. Dosyanızın formatını kontrol edin.")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


def handle_imbalance_and_train():
    df = pd.read_csv("diabetes.csv")
    st.subheader("Ham Veriler")
    st.write(df)

    # Model seçimi için ComboBox
    model_name = st.selectbox(
        "Bir Model Seçin", ["Logistic Regression", "SVC", "KNN"]
    )

    # Veri dengesizliğini çözmek için yöntem seçimi
    imbalance_method = st.selectbox(
        "Dengesizlik Yöntemi Seçin", ["İlk Hali", "SMOTE", "Random Oversampling", "Random Undersampling"]
    )

    # 1 içeren veriyi almak
    class_1 = df[df["Outcome"] == 1]

    # 0 içeren veriyi almak
    class_0 = df[df["Outcome"] == 0]

    # class_1'den 100 örnek almak (manuel seçim)
    class_1_undersampled = class_1.head(100)

    # Yeni veri setini oluşturmak
    df = pd.concat([class_0, class_1_undersampled])

    try:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Orijinal veriyi saklama
        original_X_train, original_y_train = X.copy(), y.copy()

        # Veriyi eğitim ve test olarak ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Dengesizlik çözüm yöntemi
        if imbalance_method == "İlk Hali":
            X_train, y_train = original_X_train.copy(), original_y_train.copy()
        elif imbalance_method == "SMOTE":
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        elif imbalance_method == "Random Oversampling":
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        elif imbalance_method == "Random Undersampling":
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        # İşlenmiş verileri göster
        st.subheader(f"{imbalance_method} Sonrası Veriler")
        processed_df = pd.DataFrame(X_train, columns=X.columns)
        processed_df["Outcome"] = y_train.values
        st.write(processed_df)

        # Model tanımlama ve eğitim
        model = None
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "SVC":
            model = SVC(random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier()

        if model:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Performans metrikleri
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            f1 = f1_score(y_test, y_pred, zero_division=1)

            # Karışıklık Matrisi Görselleştirme
            st.subheader(f"{imbalance_method} ile {model_name} Model Sonuçları")
            st.write("### Karışıklık Matrisi (Görsel)")
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negatif", "Pozitif"],
                yticklabels=["Negatif", "Pozitif"],
                ax=ax,
            )
            ax.set_xlabel("Tahmin Edilen")
            ax.set_ylabel("Gerçek")
            ax.set_title(f"{model_name} Karışıklık Matrisi")
            st.pyplot(fig)

            # Performans Metrikleri
            st.write("### Performans Metrikleri")
            st.write(f"- **Doğruluk (Accuracy):** {accuracy:.2f}")
            st.write(f"- **Duyarlılık (Recall):** {recall:.2f}")
            st.write(f"- **Özgüllük (Specificity):** {specificity:.2f}")
            st.write(f"- **F1-Skor:** {f1:.2f}")

        else:
            st.warning("Lütfen bir model seçin!")

    except KeyError as e:
        st.error(f"Beklenmeyen bir hata oluştu: {e}. Dosyanızın formatını kontrol edin.")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


def normalize_data_and_train():
    df = pd.read_csv("diabetes.csv")
    st.subheader("Ham Veriler")
    st.write(df)

    model_name = st.selectbox(
        "Bir Model Seçin", ["Logistic Regression", "SVC", "KNN"]
    )

    # Normalizasyon yöntemi seçimi
    normalization_method = st.selectbox(
        "Normalizasyon Yöntemi Seçin", ["Min-Max Scaling", "Standard Scaling", "Normalizasyon Yapılmamış Hali"]
    )

    try:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Veriyi eğitim ve test olarak ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Normalizasyon işlemi kontrolü
        if normalization_method == "Normalizasyon Yapılmamış Hali":
            X_train_norm = X_train
            X_test_norm = X_test
        else:
            if normalization_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
            elif normalization_method == "Standard Scaling":
                scaler = StandardScaler()

            # Eğitim ve test setlerinde normalizasyon
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)

        # Normalizasyon sonrası verileri göster
        st.subheader(f"{normalization_method} Sonrası Eğitim Verileri")
        normalized_df = pd.DataFrame(X_train_norm, columns=X.columns)
        normalized_df["Outcome"] = y_train.values
        st.write(normalized_df)

        # Model tanımlama ve eğitim
        model = None
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "SVC":
            model = SVC(random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier()

        if model:
            model.fit(X_train_norm, y_train)
            y_pred = model.predict(X_test_norm)

            # Performans metrikleri
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)  # Duyarlılık
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Özgüllük
            f1 = f1_score(y_test, y_pred, zero_division=1)

            # Karışıklık Matrisi Görselleştirme
            st.subheader(f"{normalization_method} ile {model_name} Model Sonuçları")
            st.write("### Karışıklık Matrisi (Görsel)")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"], ax=ax)
            ax.set_xlabel("Tahmin Edilen")
            ax.set_ylabel("Gerçek")
            ax.set_title(f"{model_name} Karışıklık Matrisi")
            st.pyplot(fig)

            # Performans Metrikleri
            st.write("### Performans Metrikleri")
            st.write(f"- **Doğruluk (Accuracy):** {accuracy:.2f}")
            st.write(f"- **Duyarlılık (Recall):** {recall:.2f}")
            st.write(f"- **Özgüllük (Specificity):** {specificity:.2f}")
            st.write(f"- **F1-Skor:** {f1:.2f}")
        else:
            st.warning("Lütfen bir model seçin!")

    except KeyError as e:
        st.error(f"Beklenmeyen bir hata oluştu: {e}. Dosyanızın formatını kontrol edin.")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


def handle_noisy_data_and_train():
    df = pd.read_csv("diabetes.csv")

    # Veri gösterimi
    st.subheader("Ham Veriler")
    st.write(df)

    # Model seçimi için ComboBox
    model_name = st.selectbox("Bir Model Seçin", ["Logistic Regression", "SVC", "KNN"])

    # Gürültü ekleme yüzdesini sabitledik
    noise_percentage = 15
    noisy_columns = st.multiselect("Gürültü Eklemek İstediğiniz Sütunları Seçin", df.columns)

    if noisy_columns:
        noisy_df = df.copy()
        num_noisy_samples = int(len(noisy_df) * (noise_percentage / 100))

        # Gürültü değerlerini saklamak için bir sözlük
        noisy_indices_map = {}

        for col in noisy_columns:
            noisy_indices = np.random.choice(noisy_df.index, num_noisy_samples, replace=False)
            noise_values = np.random.uniform(
                noisy_df[col].min(), noisy_df[col].max(), size=num_noisy_samples
            )
            noisy_df.loc[noisy_indices, col] = noise_values
            noisy_indices_map[col] = noisy_indices

        st.subheader("Gürültü Eklenmiş Veriler")
        st.write(noisy_df)

        # Gürültü işleme seçenekleri
        strategy = st.selectbox("Bir Gürültü İşleme Stratejisi Seçin",
                                ["Gürültü Eklendikten Sonra", "Gürültüleri Çıkarma", "Gürültüleri 0 ile Doldurma"])

        # Model tanımlama
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "SVC":
            model = SVC(random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        else:
            st.warning("Lütfen bir model seçin!")
            return

        def visualize_cm(cm, title):
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
            plt.title(title)
            plt.ylabel("Gerçek Değerler")
            plt.xlabel("Tahmin Edilen Değerler")
            st.pyplot(plt)

        def train_and_evaluate(model, data):
            X = data.drop("Outcome", axis=1)
            y = data["Outcome"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            f1 = f1_score(y_test, y_pred, zero_division=1)

            return cm, accuracy, precision, recall, specificity, f1

        if strategy == "Gürültü Eklendikten Sonra":
            st.write("### Gürültü Eklenmiş Verinin Sonuçları")
            cm_noisy, acc_noisy, prec_noisy, rec_noisy, spec_noisy, f1_noisy = train_and_evaluate(model, noisy_df)
            visualize_cm(cm_noisy, "Gürültü Eklenmiş Karışıklık Matrisi")
            st.write(f"- **Doğruluk:** {acc_noisy:.2f}")
            st.write(f"- **Duyarlılık:** {rec_noisy:.2f}")
            st.write(f"- **Özgüllük:** {spec_noisy:.2f}")
            st.write(f"- **F1-Skor:** {f1_noisy:.2f}")

        elif strategy == "Gürültüleri Çıkarma":
            cleaned_df = noisy_df.copy()
            for col, indices in noisy_indices_map.items():
                cleaned_df.loc[indices, col] = np.nan
            cleaned_df = cleaned_df.dropna()
            st.subheader("Gürültü Çıkarıldıktan Sonra Veri")
            st.write(cleaned_df)

            cm_clean, acc_clean, prec_clean, rec_clean, spec_clean, f1_clean = train_and_evaluate(model, cleaned_df)
            visualize_cm(cm_clean, "Gürültü Çıkarıldıktan Sonra Karışıklık Matrisi")
            st.write(f"- **Doğruluk:** {acc_clean:.2f}")
            st.write(f"- **Duyarlılık:** {rec_clean:.2f}")
            st.write(f"- **Özgüllük:** {spec_clean:.2f}")
            st.write(f"- **F1-Skor:** {f1_clean:.2f}")

        elif strategy == "Gürültüleri 0 ile Doldurma":
            cleaned_df = noisy_df.copy()
            for col, indices in noisy_indices_map.items():
                cleaned_df.loc[indices, col] = 0
            st.subheader("Gürültü 0 ile Doldurulduktan Sonra Veri")
            st.write(cleaned_df)

            cm_clean, acc_clean, prec_clean, rec_clean, spec_clean, f1_clean = train_and_evaluate(model, cleaned_df)
            visualize_cm(cm_clean, "Gürültü 0 ile Doldurulduktan Sonra Karışıklık Matrisi")
            st.write(f"- **Doğruluk:** {acc_clean:.2f}")
            st.write(f"- **Duyarlılık:** {rec_clean:.2f}")
            st.write(f"- **Özgüllük:** {spec_clean:.2f}")
            st.write(f"- **F1-Skor:** {f1_clean:.2f}")

    else:
        st.warning("Lütfen en az bir sütun seçin!")



def k_fold_cross_validation():
    df = pd.read_csv("diabetes.csv")

    # Veri gösterimi
    st.subheader("Ham Veriler")
    st.write(df)

    # Model seçimi için ComboBox
    model_name = st.selectbox(
        "Bir Model Seçin", ["Logistic Regression", "SVC", "KNN"]
    )

    # K-fold için kullanıcıdan değer alma
    n_splits = st.number_input("K-Fold değerini girin (2 ile 20 arasında):", min_value=2, max_value=20, value=5, step=1)

    try:
        # Özellikler ve hedef değişkeni ayırma
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # K-Fold çapraz doğrulama işlemi
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Model seçimine göre uygun modelin tanımlanması
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "SVC":
            model = SVC(random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier()

        # Performans metriklerini depolamak için listeler
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        confusion_matrices = []

        # K-Fold çapraz doğrulama işlemi
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Performans metrikleri hesaplama
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=1))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=1))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=1))
            confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Ortalama sonuçlar
        st.subheader(f"{model_name} Modeli ile K-Fold Sonuçları")
        st.write("### K-Fold Performans Metrikleri")
        st.write(f"- **Ortalama Doğruluk (Accuracy):** {np.mean(accuracy_scores):.2f}")
        st.write(f"- **Ortalama Duyarlılık (Recall):** {np.mean(recall_scores):.2f}")
        st.write(f"- **Ortalama F1-Skor:** {np.mean(f1_scores):.2f}")

        # Ortalama karışıklık matrisi
        avg_confusion_matrix = np.mean(confusion_matrices, axis=0).astype(int)
        st.write("### Ortalama Karışıklık Matrisi")
        fig, ax = plt.subplots()
        sns.heatmap(avg_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"], ax=ax)
        ax.set_xlabel("Tahmin Edilen")
        ax.set_ylabel("Gerçek")
        ax.set_title("Ortalama Karışıklık Matrisi")
        st.pyplot(fig)

        # Katlama bazında sonuçlar
        st.write("### Her Katlamadaki Performans ve Karışıklık Matrisi")
        for fold, (acc, rec, f1, cm) in enumerate(zip(accuracy_scores, recall_scores, f1_scores, confusion_matrices), 1):
            st.write(f"#### Katlama {fold}:")
            st.write(f"- **Doğruluk (Accuracy):** {acc:.2f}")
            st.write(f"- **Duyarlılık (Recall):** {rec:.2f}")
            st.write(f"- **F1-Skor:** {f1:.2f}")

            # Karışıklık Matrisi Görselleştirme
            st.write("##### Karışıklık Matrisi (Görsel):")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"], ax=ax)
            ax.set_xlabel("Tahmin Edilen")
            ax.set_ylabel("Gerçek")
            ax.set_title(f"Katlama {fold} Karışıklık Matrisi")
            st.pyplot(fig)
            st.write("---")

    except KeyError as e:
        st.error(f"Beklenmeyen bir hata oluştu: {e}. Dosyanızın formatını kontrol edin.")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


def predict_proba():
    try:
        # Veri setini yükleyin
        df = pd.read_csv("diabetes.csv")  # Verinizin yolunu doğru şekilde belirtin

        st.subheader("Ham Veriler")
        st.write(df)

        # Özellikler ve hedef değişkeni
        target = 'Outcome'  # Hedef değişken 'Outcome'
        features = [col for col in df.columns if col != target]  # Özellikler
        X = df[features]  # Özellikler (X)
        y = df[target]  # Etiketler (y)

        # Normalizasyon: Min-Max Scaler uygulama
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)  # Veriyi [0, 1] aralığına dönüştür

        # Eğitim ve test verilerine ayırma (Normalleştirilmiş veri ile)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Modelleri tanımlayın
        models = {
            "SVC": SVC(probability=True, random_state=42),  # Random Forest yerine SVC kullandık
            "Logistic Regression": LogisticRegression(random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
        }

        best_model = None
        best_accuracy = 0

        # Tüm modelleri çalıştır ve en iyi modeli seç
        for model_name, model in models.items():
            st.write(f"### {model_name} Modeli (Normalizasyon ile)")

            # Modeli eğitme
            model.fit(X_train, y_train)

            # Predict ve Predict_Proba
            y_pred = model.predict(X_test)  # Predict (sınıf tahminleri)
            y_pred_proba = model.predict_proba(X_test)  # Predict_Proba (olasılık tahminleri)

            # Performans metrikleri
            acc_scaled = accuracy_score(y_test, y_pred)  # Doğruluk hesaplama
            cm_scaled = confusion_matrix(y_test, y_pred)  # Karışıklık matrisi
            report_scaled = classification_report(y_test, y_pred, output_dict=True)

            st.write(f"Doğruluk: {acc_scaled:.2f}")
            st.write("Sınıflandırma Raporu:")
            st.json(report_scaled)

            st.write("Tahmin Olasılıkları (Predict_Proba):")
            st.write(y_pred_proba)

            # Karışıklık Matrisi
            st.write("### Karışıklık Matrisi:")
            fig, ax = plt.subplots()
            sns.heatmap(cm_scaled, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel("Tahmin Edilen")
            plt.ylabel("Gerçek")
            st.pyplot(fig)

            # En iyi modeli seçme (en yüksek doğruluk ile)
            if acc_scaled > best_accuracy:
                best_accuracy = acc_scaled
                best_model = model

        # En iyi modelin tahminlerini yapma
        st.write(f"### En İyi Model: {best_model.__class__.__name__} (Doğruluk: {best_accuracy:.2f})")
        y_pred_best = best_model.predict(X_test)
        y_pred_proba_best = best_model.predict_proba(X_test)

        # En iyi model ile tahminler
        st.write("En İyi Model ile Tahminler:")
        st.write(f"**Predict (Sınıf Tahminleri):**")
        st.write(y_pred_best)

        st.write(f"**Predict_Proba (Sınıf Olasılıkları):**")
        st.write(y_pred_proba_best)

        # Karışıklık matrisi (En iyi model)
        st.write("### En İyi Model Karışıklık Matrisi:")
        cm_best = confusion_matrix(y_test, y_pred_best)
        fig, ax = plt.subplots()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


def model_tahmin():
    Pregnancies = st.number_input("Pregnancies (Gebelik Sayısı):", min_value=0, max_value=20, value=0)
    Glucose = st.number_input("Glucose (Glikoz Düzeyi):", min_value=0, max_value=200, value=120)
    BloodPressure = st.number_input("BloodPressure (Kan Basıncı):", min_value=0, max_value=200, value=70)
    SkinThickness = st.number_input("SkinThickness (Cilt Kalınlığı):", min_value=0, max_value=100, value=20)
    Insulin = st.number_input("Insulin (İnsülin Düzeyi):", min_value=0, max_value=1000, value=80)
    BMI = st.number_input("BMI (Vücut Kitle İndeksi):", min_value=0.0, max_value=100.0, value=30.0)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction (Ailede Diyabet Geçmişi):", min_value=0.0,
                                               max_value=2.0, value=0.5)
    Age = st.number_input("Age (Yaş):", min_value=0, max_value=120, value=35)

    # Kullanıcıdan alınan verileri bir DataFrame'e dönüştürme
    new_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    # Eğitim verisini yükleyelim (bu kısmı model eğitiminin yapıldığı yer olarak düşünün)
    df = pd.read_csv('diabetes.csv')

    # Özellikler (girdi) ve hedef (çıkış)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Model ve normalizasyon işlemleri için eğitim
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modeli eğitelim
    model = LogisticRegression()
    model.fit(X_scaled, y)

    # Yeni veriyi aynı şekilde normalleştiriyoruz
    new_data_scaled = scaler.transform(new_data)

    # Model ile tahmin yapalım
    prediction = model.predict(new_data_scaled)
    prediction_proba = model.predict_proba(new_data_scaled)

    # Tahmin sonucu ve olasılıkları Streamlit uygulamasında gösterme
    if prediction[0] == 1:
        st.write(f"Tahmin Sonucu: **Diyabet var** (Outcome: 1)")
    else:
        st.write(f"Tahmin Sonucu: **Diyabet yok** (Outcome: 0)")

    st.write(f"Tahmin Olasılıkları: [Diyabet Yok, Diyabet Var]: {prediction_proba[0]}")



# Buton durumlarını tanımlama
buttons = [
    "Ham Verilerle Çalışma",
    "Dengesizlikle Başa Çıkma",
    "Normalizasyon",
    "Gürültülü Verilerle Çalışma",
    "k-Fold Çapraz Doğrulama",
    "Predict-Predict_Proba",
    "Tahmin Yaptır"
]

# Her buton için başlangıç durumu
for button in buttons:
    if f"{button}_clicked" not in st.session_state:
        st.session_state[f"{button}_clicked"] = False

# Butonların işlevselliği
if st.button("1. Ham Verilerle Çalışma"):
    st.session_state["Ham Verilerle Çalışma_clicked"] = not st.session_state["Ham Verilerle Çalışma_clicked"]

if st.button("2. Dengesizlikle Başa Çıkma"):
    st.session_state["Dengesizlikle Başa Çıkma_clicked"] = not st.session_state["Dengesizlikle Başa Çıkma_clicked"]

if st.button("3. Normalizasyon"):
    st.session_state["Normalizasyon_clicked"] = not st.session_state["Normalizasyon_clicked"]

if st.button("4. Gürültülü Verilerle Çalışma"):
    st.session_state["Gürültülü Verilerle Çalışma_clicked"] = not st.session_state["Gürültülü Verilerle Çalışma_clicked"]

if st.button("5. K-Fold Çapraz Doğrulama"):
    st.session_state["k-Fold Çapraz Doğrulama_clicked"] = not st.session_state["k-Fold Çapraz Doğrulama_clicked"]

if st.button("6. Predict-Predict_Proba"):
    st.session_state["Predict-Predict_Proba_clicked"] = not st.session_state["Predict-Predict_Proba_clicked"]

if st.button("7. Tahmin Yaptır"):
    st.session_state["Tahmin Yaptır_clicked"] = not st.session_state["Tahmin Yaptır_clicked"]

# Duruma göre işlem çağırma
if st.session_state["Ham Verilerle Çalışma_clicked"]:
    perform_ml()

if st.session_state["Dengesizlikle Başa Çıkma_clicked"]:
    handle_imbalance_and_train()

if st.session_state["Normalizasyon_clicked"]:
    normalize_data_and_train()

if st.session_state["Gürültülü Verilerle Çalışma_clicked"]:
    handle_noisy_data_and_train()

if st.session_state["k-Fold Çapraz Doğrulama_clicked"]:
    k_fold_cross_validation()

if st.session_state["Predict-Predict_Proba_clicked"]:
    predict_proba()

if st.session_state["Tahmin Yaptır_clicked"]:
    model_tahmin()