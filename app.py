import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def plot_method_comparison(df_komunitas, cbf_scores_norm, cf_scores_norm, user_idx=0, top_n=5):
    """
    Membandingkan hasil rekomendasi dari berbagai metode dalam satu grafik
    """
    # Ambil top N rekomendasi dari masing-masing metode
    # CBF Only
    cbf_indices = np.argsort(cbf_scores_norm[user_idx])[::-1][:top_n]
    cbf_communities = [df_komunitas["nama_komunitas"].iloc[i] for i in cbf_indices]
    cbf_scores = [cbf_scores_norm[user_idx, i] for i in cbf_indices]
    
    # CF Only
    cf_indices = np.argsort(cf_scores_norm[user_idx])[::-1][:top_n]
    cf_communities = [df_komunitas["nama_komunitas"].iloc[i] for i in cf_indices]
    cf_scores = [cf_scores_norm[user_idx, i] for i in cf_indices]
    
    # Simple Average
    avg_scores = (cbf_scores_norm[user_idx] + cf_scores_norm[user_idx]) / 2
    avg_indices = np.argsort(avg_scores)[::-1][:top_n]
    avg_communities = [df_komunitas["nama_komunitas"].iloc[i] for i in avg_indices]
    avg_scores_values = [avg_scores[i] for i in avg_indices]
    
    # Weighted (70% CBF, 30% CF)
    weighted_scores = 0.7 * cbf_scores_norm[user_idx] + 0.3 * cf_scores_norm[user_idx]
    weighted_indices = np.argsort(weighted_scores)[::-1][:top_n]
    weighted_communities = [df_komunitas["nama_komunitas"].iloc[i] for i in weighted_indices]
    weighted_scores_values = [weighted_scores[i] for i in weighted_indices]
    
    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perbandingan Hasil Rekomendasi dari Berbagai Metode', fontsize=16)
    
    # CBF Plot
    axs[0, 0].barh(cbf_communities, cbf_scores, color='#3498db')
    axs[0, 0].set_title('Content-Based Filtering (100% CBF)')
    axs[0, 0].set_xlim(0, 1.0)
    
    # CF Plot
    axs[0, 1].barh(cf_communities, cf_scores, color='#e74c3c')
    axs[0, 1].set_title('Collaborative Filtering (100% CF)')
    axs[0, 1].set_xlim(0, 1.0)
    
    # Simple Average Plot
    axs[1, 0].barh(avg_communities, avg_scores_values, color='#2ecc71')
    axs[1, 0].set_title('Simple Average (50% CBF + 50% CF)')
    axs[1, 0].set_xlim(0, 1.0)
    
    # Weighted Average Plot
    axs[1, 1].barh(weighted_communities, weighted_scores_values, color='#9b59b6')
    axs[1, 1].set_title('Weighted Average (70% CBF + 30% CF)')
    axs[1, 1].set_xlim(0, 1.0)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def analyze_recommendation_overlap(df_komunitas, cbf_scores_norm, cf_scores_norm, user_idx, top_n=5):
    """
    Menganalisis overlap antara rekomendasi dari metode berbeda
    """
    # Dapatkan indeks top N rekomendasi
    cbf_top = np.argsort(cbf_scores_norm[user_idx])[::-1][:top_n]
    cf_top = np.argsort(cf_scores_norm[user_idx])[::-1][:top_n]
    avg_top = np.argsort((cbf_scores_norm[user_idx] + cf_scores_norm[user_idx])/2)[::-1][:top_n]
    
    # Dapatkan set komunitas
    cbf_set = set([df_komunitas["nama_komunitas"].iloc[i] for i in cbf_top])
    cf_set = set([df_komunitas["nama_komunitas"].iloc[i] for i in cf_top])
    avg_set = set([df_komunitas["nama_komunitas"].iloc[i] for i in avg_top])
    
    # Hitung overlap
    cbf_cf_overlap = len(cbf_set.intersection(cf_set))
    avg_cbf_overlap = len(avg_set.intersection(cbf_set))
    avg_cf_overlap = len(avg_set.intersection(cf_set))
    
    # Buat analysis
    if cbf_cf_overlap == 0:
        analysis = "Kedua metode (CBF dan CF) memberikan rekomendasi yang sangat berbeda. Hal ini menunjukkan bahwa profil mahasiswa dan pola rating memiliki kecenderungan yang berbeda."
    elif cbf_cf_overlap >= 3:
        analysis = "Kedua metode (CBF dan CF) memberikan banyak rekomendasi yang sama, menunjukkan konsistensi antara profil mahasiswa dan pola rating."
    else:
        analysis = "Ada beberapa overlap antara rekomendasi CBF dan CF, yang menunjukkan bahwa kedua metode saling melengkapi."
    
    method_dominance = "CBF" if avg_cbf_overlap > avg_cf_overlap else "CF"
    dominance_analysis = f"Rekomendasi simple average lebih banyak dipengaruhi oleh metode {method_dominance}."
    
    # Return hasil analisis
    return {
        "cbf_cf_overlap": f"{cbf_cf_overlap}/{min(len(cbf_set), len(cf_set))}",
        "avg_cbf_overlap": f"{avg_cbf_overlap}/{len(avg_set)}",
        "avg_cf_overlap": f"{avg_cf_overlap}/{len(avg_set)}",
        "analysis": analysis,
        "dominance": dominance_analysis
    }

def preprocess_data_mahasiswa(df):
    """
    Preprocessing data mahasiswa
    """
    # Buat ID mahasiswa jika belum ada
    if "id_mahasiswa" not in df.columns:
        if "Mahasiswa" in df.columns:
            df["id_mahasiswa"] = df["Mahasiswa"]
        else:
            df["id_mahasiswa"] = [f"MHS_{i}" for i in range(len(df))]
    
    # Buat kolom nama mahasiswa jika belum ada
    if "nama_mahasiswa" not in df.columns:
        if "Mahasiswa" in df.columns:
            df["nama_mahasiswa"] = df["Mahasiswa"]
        else:
            df["nama_mahasiswa"] = df["id_mahasiswa"]
    
    # Bersihkan data yang tidak perlu
    df.drop(columns=["Timestamp", "Email Address"], inplace=True, errors='ignore')
    
    # Pastikan kolom profil ada
    required_cols = ["passion", "pengetahuan_sebelumnya", "tim", "skill", "motivasi"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Buat profil gabungan
    df["profil_mahasiswa"] = df[required_cols].fillna("").agg(" ".join, axis=1)
    
    return df

def preprocess_data_komunitas(df):
    """
    Preprocessing data komunitas
    """
    # Buat ID komunitas jika belum ada
    if "id_komunitas" not in df.columns:
        df["id_komunitas"] = [f"KOM_{i}" for i in range(len(df))]
    
    # Buat kategori jika belum ada
    if "kategori" not in df.columns:
        df["kategori"] = "Umum"
    
    # Pastikan kolom konten ada
    required_kom_cols = ["nama_komunitas", "deskripsi", "aktivitas", "teknologi", "visi_misi"]
    for col in required_kom_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Buat konten gabungan
    df["konten_komunitas"] = df[required_kom_cols].fillna("").agg(" ".join, axis=1)
    
    return df

def standardize_rating_columns(df_mahasiswa, df_komunitas, original_rating_cols):
    """
    Standardisasi kolom rating dan memastikan konsistensi dengan komunitas
    """
    # Extract nama komunitas dari kolom rating
    rating_to_komunitas = {}
    for col in original_rating_cols:
        if '[' in col and ']' in col:
            kom_name = col.split('[')[1].split(']')[0].strip()
            rating_to_komunitas[col] = kom_name
    
    # Set komunitas unik
    unique_komunitas = list(set(rating_to_komunitas.values()))
    
    # Pastikan semua komunitas ada di dataframe komunitas
    for kom in unique_komunitas:
        if kom not in df_komunitas["id_komunitas"].values:
            if kom not in df_komunitas["nama_komunitas"].values:
                # Tambahkan komunitas baru
                new_row = {
                    "id_komunitas": kom,
                    "nama_komunitas": kom,
                    "deskripsi": f"Komunitas {kom}",
                    "kategori": "Umum",
                    "konten_komunitas": f"Komunitas {kom}"
                }
                df_komunitas = pd.concat([df_komunitas, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Gunakan nama komunitas sebagai ID
                kom_idx = df_komunitas[df_komunitas["nama_komunitas"] == kom].index[0]
                df_komunitas.loc[kom_idx, "id_komunitas"] = kom
    
    # Update list ID komunitas
    kom_ids = df_komunitas["id_komunitas"].tolist()
    
    # Buat kolom rating standar
    standardized_rating_cols = []
    
    # Map kolom rating yang ada ke format standar
    for orig_col, kom_name in rating_to_komunitas.items():
        if kom_name in kom_ids:
            std_col = f"Rating_{kom_name}"
            df_mahasiswa[std_col] = df_mahasiswa[orig_col]
            standardized_rating_cols.append(std_col)
    
    # Tambahkan kolom rating yang belum ada
    for kom_id in kom_ids:
        std_col = f"Rating_{kom_id}"
        if std_col not in df_mahasiswa.columns:
            df_mahasiswa[std_col] = 0.0
            standardized_rating_cols.append(std_col)
    
    return df_mahasiswa, df_komunitas, standardized_rating_cols

# ======================================================
# APP CONFIGURATION
# ======================================================

# Set page configuration
st.set_page_config(page_title="Sistem Rekomendasi Akademik", layout="wide")

# Title and description
st.title("🎓 Sistem Rekomendasi Komunitas Akademik (Mixed Hybrid Filtering)")
st.markdown("""
Sistem ini menggunakan kombinasi Content-Based Filtering (CBF) dan Collaborative Filtering (CF) 
untuk memberikan rekomendasi komunitas akademik yang sesuai dengan profil dan preferensi mahasiswa.
""")

# ======================================================
# SIDEBAR CONFIGURATION
# ======================================================

with st.sidebar:
    st.header("Pengaturan Sistem Rekomendasi")
    
    # Metode input data
    input_mode = st.radio("Pilih metode input data:", ["Upload CSV", "Input Manual"])
    
    # Metode hybrid
    st.header("Pengaturan Hybrid Filtering")
    hybrid_method = st.radio(
        "Pilih metode penggabungan:",
        ["Simple Average (Rata-rata)", "Weighted Average (Pembobotan)"]
    )
    
    if hybrid_method == "Weighted Average (Pembobotan)":
        alpha = st.slider(
            "Bobot α (CBF vs CF)", 
            0.0, 1.0, 0.5, 0.05,
            help="α = 1 berarti hanya menggunakan Content-Based Filtering, α = 0 berarti hanya menggunakan Collaborative Filtering"
        )
    else:
        st.info("Metode Simple Average memberikan bobot yang sama (0.5) untuk CBF dan CF")
        alpha = 0.5  # Default untuk simple average
    
    # Parameter lainnya
    top_n = st.slider("Top-N rekomendasi", 1, 10, 5)
    show_details = st.checkbox("Tampilkan Detail Skor", value=True)
    show_calc_steps = st.checkbox("Tampilkan Langkah Perhitungan", value=True)

# ======================================================
# DATA LOADING AND PREPROCESSING
# ======================================================

# Inisialisasi variabel dataframe
df_mahasiswa = None
df_komunitas = None

# Load data sesuai mode input
if input_mode == "Upload CSV":
    st.subheader("📂 Upload File CSV")
    
    col1, col2 = st.columns(2)
    with col1:
        file_mhs = st.file_uploader("Upload file *synthetic_data_mahasiswa.csv*", type="csv")
    with col2:
        file_kom = st.file_uploader("Upload file *data_komunitas.csv*", type="csv")
    
    if file_mhs and file_kom:
        # Load data
        df_mahasiswa = pd.read_csv(file_mhs)
        df_komunitas = pd.read_csv(file_kom)
        
        # Tampilkan preview data
        with st.expander("Preview Data"):
            st.write("### Preview Data Mahasiswa")
            st.dataframe(df_mahasiswa.head())
            
            st.write("### Preview Data Komunitas")
            st.dataframe(df_komunitas.head())
            
            # Tampilkan kolom rating
            rating_columns = [col for col in df_mahasiswa.columns if 'Rating' in col]
            st.write("### Kolom Rating yang Ditemukan:")
            st.write(rating_columns)
        
        # Preprocessing
        df_mahasiswa = preprocess_data_mahasiswa(df_mahasiswa)
        df_komunitas = preprocess_data_komunitas(df_komunitas)

elif input_mode == "Input Manual":
    st.subheader("✍️ Input Data Manual")
    
    # Default komunitas untuk input manual
    default_komunitas = ["GDSC", "DSE", "UINUX", "ETH0", "WEBONDER", "MOCAP", "ONTAKI", "FUN JAVA", "UINBUNTU", "MAMUD"]

# Informasi tentang komunitas (dengan icon)
    komunitas_info = {
    "GDSC": {"desc": "Google Developer Student Clubs", "icon": "🌐", "color": "#4285F4"},
    "DSE": {"desc": "Data Science Enthusiast", "icon": "📊", "color": "#0F9D58"},
    "UINUX": {"desc": "Komunitas Linux dan Open Source", "icon": "🐧", "color": "#FFA500"},
    "ETH0": {"desc": "Komunitas Cyber Security", "icon": "🔒", "color": "#FF4500"},
    "WEBONDER": {"desc": "Komunitas Web Development", "icon": "🌍", "color": "#800080"},
    "MOCAP": {"desc": "Komunitas Mobile Capture", "icon": "📱", "color": "#1E90FF"},
    "ONTAKI": {"desc": "Komunitas Teknologi Robotics", "icon": "🤖", "color": "#FFD700"},
    "FUN JAVA": {"desc": "Komunitas Java Programming", "icon": "☕", "color": "#8B4513"},
    "UINBUNTU": {"desc": "Komunitas Ubuntu dan Linux", "icon": "💿", "color": "#FF6347"},
    "MAMUD": {"desc": "Komunitas Manusia Multimedia", "icon": "🎨", "color": "#32CD32"}
    }

# CSS untuk card style
    st.markdown("""
    <style>
    .komunitas-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }
    .komunitas-card:hover {
        transform: translateY(-5px);
    }
    .komunitas-header {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 5px;
    }
    .komunitas-desc {
        color: #666;
        font-size: 0.9em;
    }
    .komunitas-icon {
        font-size: 2em;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Form input untuk mahasiswa
    with st.form("input_mahasiswa"):
    # Bagian profil mahasiswa
        st.subheader("📝 Profil Mahasiswa")
        col1, col2 = st.columns(2)
    
        with col1:
            nama_mhs = st.text_input("Nama Mahasiswa")
            passion = st.text_input("Passion", placeholder="Contoh: Mobile Development, Data Science, UI/UX")
            pengetahuan = st.text_input("Pengetahuan Sebelumnya", placeholder="Contoh: Python, Flutter, React")
    
        with col2:
            tim = st.text_input("Gaya Kerja Tim", placeholder="Contoh: Kolaboratif, Individual, Leadership")
            skill = st.text_input("Skill Teknis", placeholder="Contoh: Java, Database, Git")
            motivasi = st.text_input("Motivasi", placeholder="Contoh: Karir, Project, Networking")
    
    # Bagian rating komunitas
        st.markdown("---")
        st.subheader("⭐ Rating Komunitas")
        st.write("Berikan rating (0-5) untuk komunitas berikut berdasarkan minat Anda:")
    
    # Bagi komunitas menjadi 2 baris
        row1_kom = default_komunitas[:5]  # 5 komunitas pertama
        row2_kom = default_komunitas[5:]  # 5 komunitas terakhir
    
        rating_cols = {}
    
    # Baris 1 komunitas
        st.markdown("#### Baris 1")
        cols1 = st.columns(5)
        for i, komunitas in enumerate(row1_kom):
            with cols1[i]:
            # Card style dengan HTML
                color = komunitas_info[komunitas]["color"]
                icon = komunitas_info[komunitas]["icon"]
                desc = komunitas_info[komunitas]["desc"]
            
                st.markdown(f"""
                <div class="komunitas-card" style="border-left: 5px solid {color};">
                    <div class="komunitas-icon">{icon}</div>
                    <div class="komunitas-header">{komunitas}</div>
                    <div class="komunitas-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            
                rating_cols[komunitas] = st.slider(
                 label=f"Rating {komunitas}", 
                    min_value=0.0, 
                    max_value=5.0, 
                    value=0.0, 
                    step=0.5,
                    key=f"rating_{komunitas}"
                )
    
    # Baris 2 komunitas
        st.markdown("#### Baris 2")
        cols2 = st.columns(5)
        for i, komunitas in enumerate(row2_kom):
            with cols2[i]:
            # Card style dengan HTML
                color = komunitas_info[komunitas]["color"]
                icon = komunitas_info[komunitas]["icon"]
                desc = komunitas_info[komunitas]["desc"]
            
                st.markdown(f"""
                <div class="komunitas-card" style="border-left: 5px solid {color};">
                    <div class="komunitas-icon">{icon}</div>
                    <div class="komunitas-header">{komunitas}</div>
                    <div class="komunitas-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            
                rating_cols[komunitas] = st.slider(
                    label=f"Rating {komunitas}", 
                    min_value=0.0, 
                    max_value=5.0, 
                    value=0.0, 
                    step=0.5,
                    key=f"rating_{komunitas}"
                )
    
    # Submit button dengan styling
        st.markdown("---")
        col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
        with col_submit2:
            submitted = st.form_submit_button("💾 Simpan Data", use_container_width=True)

# Jika form disubmit
    if submitted and 'nama_mhs' in locals() and nama_mhs:
        profil_mahasiswa = f"{passion} {pengetahuan} {tim} {skill} {motivasi}"
    
    # Membuat dataframe mahasiswa
        mahasiswa_data = {
            "Mahasiswa": [nama_mhs],
            "nama_mahasiswa": [nama_mhs],
            "id_mahasiswa": ["MHS_1"],
            "profil_mahasiswa": [profil_mahasiswa],
            "passion": [passion],
            "pengetahuan_sebelumnya": [pengetahuan],
            "tim": [tim],
            "skill": [skill],
            "motivasi": [motivasi]
        }
    
    # Menambahkan data rating ke dataframe
        for kom, rating in rating_cols.items():
            mahasiswa_data[f"Rating [{kom}]"] = [rating]
    
        df_mahasiswa = pd.DataFrame(mahasiswa_data)
    
    # Tampilkan ringkasan data
        st.success(f"✅ Data untuk {nama_mhs} berhasil diinput!")
    
    # Tampilkan visualisasi rating
        st.subheader("📊 Ringkasan Rating")
    
        rating_data = {
            'Komunitas': list(rating_cols.keys()),
            'Rating': list(rating_cols.values())
        }
        rating_df = pd.DataFrame(rating_data)
        rating_df = rating_df.sort_values('Rating', ascending=False)
    
    # Hanya tampilkan komunitas dengan rating > 0
        non_zero_ratings = rating_df[rating_df['Rating'] > 0]
    
        if len(non_zero_ratings) > 0:
        # Plot bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(non_zero_ratings['Komunitas'], non_zero_ratings['Rating'], 
                    color=[komunitas_info[k]['color'] for k in non_zero_ratings['Komunitas']])
        
            ax.set_xlabel('Rating')
            ax.set_title('Rating Komunitas')
            ax.set_xlim(0, 5)
        
        # Tambahkan label rating
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                        ha='left', va='center')
        
            st.pyplot(fig)
        else:
            st.info("Belum ada rating yang diberikan untuk komunitas manapun.")
    
    # Jika form disubmit dan ada nama mahasiswa
        if submitted and nama_mhs:
        # Gabungkan profil mahasiswa
            profil_mahasiswa = f"{passion} {pengetahuan} {tim} {skill} {motivasi}"
        
        # Siapkan data mahasiswa termasuk rating
            mahasiswa_data = {
                "Mahasiswa": [nama_mhs],
                "nama_mahasiswa": [nama_mhs],
                "id_mahasiswa": ["MHS_1"],
                "profil_mahasiswa": [profil_mahasiswa],
                "passion": [passion],
                "pengetahuan_sebelumnya": [pengetahuan],
                "tim": [tim],
                "skill": [skill],
                "motivasi": [motivasi]
            }
        
        # Tambahkan kolom rating dengan format yang sesuai dengan data CSV
            for komunitas, rating in rating_cols.items():
                rating_col_name = f"Rating [{komunitas}]"
                mahasiswa_data[rating_col_name] = [rating]
        
        # Buat DataFrame mahasiswa
            df_mahasiswa = pd.DataFrame(mahasiswa_data)
        
        # Buat DataFrame komunitas default
            # Buat DataFrame komunitas default
            df_komunitas = pd.DataFrame({
                "id_komunitas": default_komunitas,
                "nama_komunitas": default_komunitas,
                "deskripsi": [
                    "Google Developer Student Clubs - Komunitas pengembang yang didukung Google untuk mahasiswa",
                    "Data Science Enthusiast - Komunitas untuk para penggemar ilmu data dan analitik",
                    "UINUX - Komunitas pengguna dan pengembang Linux dan perangkat lunak sumber terbuka",
                    "ETH0 - Komunitas keamanan siber dan ethical hacking",
                    "WEBONDER - Komunitas pengembangan web dan UI/UX",
                    "MOCAP - Komunitas pengembangan aplikasi mobile dan teknologi capture",
                    "ONTAKI - Komunitas robotika dan teknologi otomasi",
                    "FUN JAVA - Komunitas pemrograman Java dan pengembangan aplikasi",
                    "UINBUNTU - Komunitas Ubuntu dan pengembangan sistem operasi Linux",
                    "MAMUD - Komunitas multimedia dan desain digital"
                ],
                "aktivitas": [
                    "Workshop, Hackathon, Study Jam, Webinar",
                    "Bootcamp, Datathon, Pelatihan, Proyek Kolaboratif",
                    "Install Fest, Ngoprek Linux, Workshop Command Line, Kontribusi Open Source",
                    "CTF Competitions, Security Workshop, Penetration Testing",
                    "Web Development Workshop, UI/UX Challenge, Frontend Bootcamp",
                    "Mobile App Development, Camera Tech Workshop, AR/VR Exploration",
                    "Robot Building, IoT Projects, Automation Challenge",
                    "Java Coding Camp, Object-Oriented Programming Workshop, Enterprise App Development",
                    "Linux Installation Party, System Administration Workshop, OS Development",
                    "Multimedia Production, Digital Art Workshop, Video Editing Challenge"
                ],
                "teknologi": [
                    "Android, Flutter, Firebase, Web, Cloud",
                    "Python, R, TensorFlow, Pandas, Jupyter, SQL",
                    "Ubuntu, Fedora, Git, Command Line, Shell Script",
                    "Kali Linux, Metasploit, Wireshark, OWASP Tools",
                    "HTML/CSS, JavaScript, React, Angular, Vue.js",
                    "Swift, Kotlin, React Native, Flutter, AR Kit",
                    "Arduino, Raspberry Pi, Sensors, Automation Tools",
                    "Java, Spring, Hibernate, Maven, JUnit",
                    "Ubuntu, Linux Kernel, Bash, System Administration Tools",
                    "Adobe Creative Suite, Blender, DaVinci Resolve, GIMP"
                ],
                "visi_misi": [
                    "Membangun komunitas teknologi yang inklusif dan memberdayakan mahasiswa dalam pengembangan aplikasi",
                    "Mengembangkan keterampilan data science dan menciptakan solusi berbasis data untuk masalah di sekitar",
                    "Memperkenalkan dan mempromosikan penggunaan perangkat lunak bebas dan sumber terbuka",
                    "Meningkatkan kesadaran dan keterampilan keamanan siber untuk keamanan digital",
                    "Mengembangkan kemampuan pembuatan web yang menarik dan interaktif",
                    "Mendorong inovasi dalam pengembangan aplikasi mobile dan teknologi capture",
                    "Memajukan pengetahuan di bidang robotika dan otomasi untuk solusi masa depan",
                    "Memperdalam pemahaman pemrograman Java dan pengembangan aplikasi enterprise",
                    "Mempromosikan dan mengembangkan ekosistem Linux/Ubuntu di kalangan mahasiswa",
                    "Mengeksplorasi kreativitas melalui teknologi multimedia dan desain"
                ],
                "kategori": [
                    "Teknologi", 
                    "Data Science", 
                    "Teknologi",
                    "Keamanan",
                    "Web",
                    "Mobile",
                    "Robotika",
                    "Pemrograman",
                    "Sistem Operasi",
                    "Multimedia"
                ]
            })
        
        # Buat kolom konten komunitas
            df_komunitas["konten_komunitas"] = df_komunitas[
                ["nama_komunitas", "deskripsi", "aktivitas", "teknologi", "visi_misi"]
            ].fillna("").agg(" ".join, axis=1)
        
        # Tampilkan preview data yang diinput
            st.success(f"Data untuk {nama_mhs} berhasil diinput!")
        
            with st.expander("Detail Data yang Diinput"):
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("**Profil Mahasiswa:**")
                    st.dataframe(df_mahasiswa[["nama_mahasiswa", "profil_mahasiswa"]])
            
                with col2:
                    st.write("**Rating yang Diberikan:**")
                # Buat DataFrame untuk tampilan rating
                    rating_data = {
                        'Komunitas': list(rating_cols.keys()),
                        'Rating': list(rating_cols.values())
                    }
                    rating_df = pd.DataFrame(rating_data)
                
                # Tampilkan rating dalam tabel
                    st.dataframe(rating_df)
                
                # Tambahkan visualisasi bar chart untuk rating
                    if any(rating > 0 for rating in rating_cols.values()):
                        fig, ax = plt.subplots(figsize=(8, 3))
                        sns.barplot(
                            x='Komunitas', 
                            y='Rating', 
                            data=rating_df,
                            palette='viridis',
                            ax=ax
                        )
                        ax.set_title(f'Rating dari {nama_mhs}')
                        ax.set_ylim(0, 5)
                        plt.tight_layout()
                        st.pyplot(fig)

# ======================================================
# RECOMMENDATION SYSTEM ALGORITHM
# ======================================================

# Proses rekomendasi jika data sudah ada
if df_mahasiswa is not None and df_komunitas is not None:
    st.header("🔄 Proses Rekomendasi Hybrid")
    
    with st.container():
        # === 1. CONTENT-BASED FILTERING (CBF) ===
        st.subheader("1️⃣ Content-Based Filtering (CBF)")
        
        # Tampilkan informasi input jika diminta
        if show_calc_steps:
            with st.expander("Data Input untuk CBF"):
                st.write("**Profil Mahasiswa (Sample):**")
                st.dataframe(df_mahasiswa[["nama_mahasiswa", "profil_mahasiswa"]].head())
                
                st.write("**Profil Komunitas (Sample):**")
                st.dataframe(df_komunitas[["nama_komunitas", "konten_komunitas"]].head())
        
        # TF-IDF dan perhitungan similarity
        with st.spinner("Menghitung TF-IDF dan similarity..."):
            tfidf = TfidfVectorizer()
            tfidf_kom = tfidf.fit_transform(df_komunitas["konten_komunitas"])
            tfidf_mhs = tfidf.transform(df_mahasiswa["profil_mahasiswa"])
            cbf_scores = cosine_similarity(tfidf_mhs, tfidf_kom)
        
        st.success("✅ Perhitungan Content-Based Filtering selesai!")
        
        # Tampilkan detail TF-IDF jika diminta
        if show_calc_steps:
            with st.expander("Detail TF-IDF dan Similarity"):
                st.write(f"Jumlah fitur TF-IDF: {len(tfidf.get_feature_names_out())}")
                
                # Tampilkan sample TF-IDF matrix
                tfidf_df_kom = pd.DataFrame(
                    tfidf_kom.toarray(), 
                    columns=tfidf.get_feature_names_out(),
                    index=df_komunitas["nama_komunitas"]
                )
                st.write("**Sample TF-IDF Matrix Komunitas (5 fitur pertama):**")
                st.dataframe(tfidf_df_kom.iloc[:, :5])
                
                # Visualize CBF scores
                cbf_df = pd.DataFrame(
                    cbf_scores, 
                    columns=df_komunitas["nama_komunitas"],
                    index=df_mahasiswa["nama_mahasiswa"]
                )
                st.write("**Sample CBF Scores (5 mahasiswa pertama, semua komunitas):**")
                st.dataframe(cbf_df.head())
                
                # Visualize as heatmap
                st.write("**Heatmap CBF Scores (5 mahasiswa pertama):**")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(cbf_df.head(), annot=True, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
        
        # === 2. COLLABORATIVE FILTERING (CF) ===
        st.subheader("2️⃣ Collaborative Filtering (CF)")
        
        # Identifikasi kolom rating
        original_rating_cols = [col for col in df_mahasiswa.columns if 'Rating' in col]
        
        # Tampilkan statistik rating jika diminta
        if show_calc_steps:
            with st.expander("Statistik Rating"):
                st.write("**Kolom Rating yang Ditemukan:**")
                st.write(original_rating_cols)
                
                if len(original_rating_cols) > 0:
                    rating_stats = df_mahasiswa[original_rating_cols].describe().T
                    rating_stats['non_zero_count'] = df_mahasiswa[original_rating_cols].astype(bool).sum()
                    rating_stats['non_zero_percentage'] = (rating_stats['non_zero_count'] / len(df_mahasiswa) * 100).round(2)
                    st.write("**Statistik Rating:**")
                    st.dataframe(rating_stats)
                else:
                    st.warning("Tidak ada kolom rating yang ditemukan.")
        
        # Standardisasi kolom rating
        with st.spinner("Standardisasi kolom rating..."):
            df_mahasiswa, df_komunitas, standardized_rating_cols = standardize_rating_columns(
                df_mahasiswa, df_komunitas, original_rating_cols
            )
        
        # Tampilkan detail standardisasi rating jika diminta
        if show_calc_steps:
            with st.expander("Detail Standardisasi Rating"):
                st.write("**Standardized Rating Columns:**")
                st.write(standardized_rating_cols)
                
                # Show a sample of the standardized rating data
                if len(standardized_rating_cols) > 0:
                    sample_ratings = df_mahasiswa[["nama_mahasiswa"] + standardized_rating_cols].head()
                    st.write("**Sample Standardized Ratings (5 mahasiswa pertama):**")
                    st.dataframe(sample_ratings)
        
        # Buat rating matrix
        rating_matrix = df_mahasiswa[standardized_rating_cols].copy() if standardized_rating_cols else pd.DataFrame()
        if not rating_matrix.empty:
            rating_matrix.index = df_mahasiswa["id_mahasiswa"]
            rating_matrix = rating_matrix.fillna(0).astype(float)
            
            # Tampilkan rating matrix jika diminta
            if show_calc_steps:
                with st.expander("Rating Matrix"):
                    st.write(f"Shape Rating Matrix: {rating_matrix.shape}")
                    st.write("**Sample Rating Matrix (5 mahasiswa pertama):**")
                    st.dataframe(rating_matrix.head())
                    
                    # Visualize rating distribution
                    st.write("**Distribusi Rating:**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Gather all non-zero ratings
                    all_ratings = rating_matrix.values.flatten()
                    all_ratings = all_ratings[all_ratings > 0]
                    
                    if len(all_ratings) > 0:
                        sns.histplot(all_ratings, bins=10, ax=ax)
                        ax.set_xlabel("Rating Value")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Distribution of Non-Zero Ratings")
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada rating non-zero yang ditemukan.")
            
            # Hitung similarity antar mahasiswa
            with st.spinner("Menghitung similarity antar mahasiswa..."):
                # Inisialisasi CF scores dengan ukuran yang sama dengan CBF scores
                cf_scores = np.zeros_like(cbf_scores)
                
                # Hitung user similarity jika ada minimal 2 mahasiswa
                if len(df_mahasiswa) > 1:
                    user_sim = cosine_similarity(rating_matrix)
                    
                    # Tampilkan user similarity jika diminta
                    if show_calc_steps:
                        with st.expander("User Similarity Matrix"):
                            user_sim_df = pd.DataFrame(
                                user_sim, 
                                index=df_mahasiswa["nama_mahasiswa"],
                                columns=df_mahasiswa["nama_mahasiswa"]
                            )
                            st.write("**Sample User Similarity Matrix (5x5):**")
                            st.dataframe(user_sim_df.iloc[:5, :5])
                            
                            # Visualize as heatmap
                            st.write("**Heatmap User Similarity (5x5):**")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(user_sim_df.iloc[:5, :5], annot=True, cmap="YlGnBu", ax=ax)
                            st.pyplot(fig)
                    
                    # Prediksi rating menggunakan weighted sum algorithm
                    sample_user_idx = 0
                    sample_user_cf_steps = {
                        "user": "",
                        "similarity": {},
                        "denominator": 0,
                        "weighted_ratings": {}
                    }
                    
                    kom_ids = df_komunitas["id_komunitas"].tolist()
                    
                    for i, user in enumerate(rating_matrix.index):
                        sims = user_sim[i].copy()
                        sims[i] = 0  # Exclude self similarity
                        denom = sims.sum() + 1e-6  # Avoid division by zero
                        
                        # Matrix multiplication untuk weighted sum algorithm
                        weighted_ratings = sims @ rating_matrix.values
                        cf_scores_user = weighted_ratings / denom
                        
                        # Map these scores back to our komunitas indices
                        for j, kom_id in enumerate(kom_ids):
                            std_col = f"Rating_{kom_id}"
                            if std_col in standardized_rating_cols:
                                col_idx = standardized_rating_cols.index(std_col)
                                cf_scores[i, j] = cf_scores_user[col_idx]
                        
                        # Track calculation steps for a sample user
                        if i == sample_user_idx and show_calc_steps:
                            sample_user_name = df_mahasiswa.loc[df_mahasiswa["id_mahasiswa"] == user, "nama_mahasiswa"].iloc[0]
                            sample_user_cf_steps = {
                                "user": sample_user_name,
                                "similarity": {df_mahasiswa.loc[df_mahasiswa["id_mahasiswa"] == idx, "nama_mahasiswa"].iloc[0]: sim 
                                              for idx, sim in zip(rating_matrix.index, sims) if sim > 0},
                                "denominator": denom,
                                "weighted_ratings": {df_komunitas.iloc[j]["nama_komunitas"]: cf_scores[i, j] 
                                                   for j in range(len(df_komunitas))}
                            }
                else:
                    # Untuk input manual 1 mahasiswa - gunakan rating asli sebagai CF scores
                    st.info("Mode input manual dengan satu mahasiswa. CF akan menggunakan rating yang Anda berikan.")
                    
                    # Normalisasi rating untuk CF scores (0-5 menjadi 0-1)
                    for i, user in enumerate(df_mahasiswa["id_mahasiswa"]):
                        for j, kom_id in enumerate(df_komunitas["id_komunitas"]):
                            # Cari rating kolom yang sesuai dengan komunitas
                            for col in original_rating_cols:
                                if kom_id in col:
                                    # Normalisasi nilai rating (dari 0-5 menjadi 0-1)
                                    rating_val = df_mahasiswa.iloc[i][col] / 5.0
                                    cf_scores[i, j] = rating_val
            
            st.success("✅ Perhitungan Collaborative Filtering selesai!")
            
            # Tampilkan sample calculation jika diminta
            if show_calc_steps and sample_user_cf_steps:
                with st.expander("Sample CF Calculation Steps"):
                    st.write(f"**User: {sample_user_cf_steps['user']}**")
                    
                    st.write("**Similarity dengan Pengguna Lain:**")
                    sim_df = pd.DataFrame(list(sample_user_cf_steps["similarity"].items()), 
                                      columns=["User", "Similarity"]).sort_values("Similarity", ascending=False)
                    st.dataframe(sim_df)
                    
                    st.write(f"**Denominator (sum of similarities): {sample_user_cf_steps['denominator']:.4f}**")
                    
                    st.write("**Weighted Ratings (Predicted CF Scores):**")
                    cf_pred_df = pd.DataFrame(list(sample_user_cf_steps["weighted_ratings"].items()), 
                                          columns=["Komunitas", "Predicted Score"]).sort_values("Predicted Score", ascending=False)
                    st.dataframe(cf_pred_df)
            
            # Visualize CF scores if requested
            if show_calc_steps:
                with st.expander("Final CF Scores"):
                    cf_df = pd.DataFrame(
                        cf_scores, 
                        columns=df_komunitas["nama_komunitas"],
                        index=df_mahasiswa["nama_mahasiswa"]
                    )
                    st.write("**Sample CF Scores (5 mahasiswa pertama):**")
                    st.dataframe(cf_df.head())
                    
                    # Visualize as heatmap
                    st.write("**Heatmap CF Scores (5 mahasiswa pertama):**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.heatmap(cf_df.head(), annot=True, cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
        else:
            st.warning("Tidak ada kolom rating yang ditemukan. Collaborative Filtering akan menggunakan nilai 0.")
            cf_scores = np.zeros_like(cbf_scores)
        
        # === 3. NORMALISASI SCORES ===
        st.subheader("3️⃣ Normalisasi Min-Max Scaling")
        
        with st.spinner("Menormalisasi skor..."):
            # Check if there are valid scores to normalize
            cbf_has_nonzero = np.any(cbf_scores != 0)
            cf_has_nonzero = np.any(cf_scores != 0)
            
            # Inisialisasi scaler
            cbf_scaler = MinMaxScaler()
            cf_scaler = MinMaxScaler()
            
            # Reshape untuk scaler (karena MinMaxScaler bekerja pada 2D arrays)
            if cbf_has_nonzero:
                cbf_scores_reshaped = cbf_scores.reshape(-1, 1)
                cbf_scores_norm = cbf_scaler.fit_transform(cbf_scores_reshaped).reshape(cbf_scores.shape)
            else:
                st.warning("CBF scores are all zero. Skipping normalization for CBF.")
                cbf_scores_norm = cbf_scores.copy()
            
            if cf_has_nonzero:
                cf_scores_reshaped = cf_scores.reshape(-1, 1)
                cf_scores_norm = cf_scaler.fit_transform(cf_scores_reshaped).reshape(cf_scores.shape)
            else:
                st.warning("CF scores are all zero. Skipping normalization for CF.")
                cf_scores_norm = cf_scores.copy()
        
        st.success("✅ Normalisasi skor selesai!")
        
        # Tampilkan detail normalisasi jika diminta
        if show_calc_steps:
            with st.expander("Detail Normalisasi"):
                # For CBF
                st.write("**CBF Scores:**")
                cbf_norm_sample = pd.DataFrame({
                    "Original": cbf_scores[0, :5],
                    "Normalized": cbf_scores_norm[0, :5]
                }, index=df_komunitas["nama_komunitas"][:5])
                st.dataframe(cbf_norm_sample)
                
                # For CF
                st.write("**CF Scores:**")
                cf_norm_sample = pd.DataFrame({
                    "Original": cf_scores[0, :5],
                    "Normalized": cf_scores_norm[0, :5]
                }, index=df_komunitas["nama_komunitas"][:5])
                st.dataframe(cf_norm_sample)
        
        # === 4. HYBRID SCORE ===
        st.subheader("4️⃣ Hybrid Scoring")
        
        with st.spinner("Menghitung skor hybrid..."):
            if hybrid_method == "Weighted Average (Pembobotan)":
                st.write(f"Menggunakan **Weighted Average** dengan α = {alpha}")
                hybrid_scores = alpha * cbf_scores_norm + (1 - alpha) * cf_scores_norm
            else:  # Simple Average
                st.write("Menggunakan **Simple Average** (rata-rata sederhana)")
                hybrid_scores = (cbf_scores_norm + cf_scores_norm) / 2
        
        st.success("✅ Perhitungan skor hybrid selesai!")
        
        # Tampilkan detail hybrid calculation jika diminta
        if show_calc_steps:
            with st.expander("Detail Hybrid Calculation"):
                method_name = "Weighted Average" if hybrid_method == "Weighted Average (Pembobotan)" else "Simple Average"
                
                st.write(f"**Sample {method_name} Calculation**")
                if hybrid_method == "Weighted Average (Pembobotan)":
                    hybrid_calc_sample = pd.DataFrame({
                        "CBF (Normalized)": cbf_scores_norm[0, :5],
                        "CF (Normalized)": cf_scores_norm[0, :5],
                        f"α × CBF": [alpha * score for score in cbf_scores_norm[0, :5]],
                        f"(1-α) × CF": [(1-alpha) * score for score in cf_scores_norm[0, :5]],
                        f"Hybrid (α={alpha})": hybrid_scores[0, :5]
                    }, index=df_komunitas["nama_komunitas"][:5])
                else:  # Simple Average
                    hybrid_calc_sample = pd.DataFrame({
                        "CBF (Normalized)": cbf_scores_norm[0, :5],
                        "CF (Normalized)": cf_scores_norm[0, :5],
                        "CBF + CF": [cbf + cf for cbf, cf in zip(cbf_scores_norm[0, :5], cf_scores_norm[0, :5])],
                        "Hybrid (rata-rata)": hybrid_scores[0, :5]
                    }, index=df_komunitas["nama_komunitas"][:5])
                
                st.dataframe(hybrid_calc_sample)
                
                # Visualize hybrid composition
                st.write("**Kontribusi Masing-masing Metode pada Skor Hybrid:**")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Get top 5 communities for the sample user
                top_indices = np.argsort(hybrid_scores[0, :])[::-1][:5]
                top_communities = [df_komunitas["nama_komunitas"].iloc[i] for i in top_indices]
                
                # Create data for stacked bar chart
                if hybrid_method == "Weighted Average (Pembobotan)":
                    cbf_contribution = [alpha * cbf_scores_norm[0, i] for i in top_indices]
                    cf_contribution = [(1 - alpha) * cf_scores_norm[0, i] for i in top_indices]
                else:  # Simple Average
                    cbf_contribution = [0.5 * cbf_scores_norm[0, i] for i in top_indices]
                    cf_contribution = [0.5 * cf_scores_norm[0, i] for i in top_indices]
                
                # Create the stacked bar chart
                ax.bar(top_communities, cbf_contribution, label='CBF Contribution')
                ax.bar(top_communities, cf_contribution, bottom=cbf_contribution, label='CF Contribution')
                
                ax.set_ylabel('Score Contribution')
                ax.set_title(f'Hybrid Score Composition ({method_name})')
                ax.legend()
                
                # Rotate labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
        
        # === 5. GENERATE RECOMMENDATIONS ===
        st.subheader("5️⃣ Hasil Rekomendasi")
        
        with st.spinner("Membuat rekomendasi final..."):
            rekos = []
            for i, mhs_id in enumerate(df_mahasiswa["id_mahasiswa"]):
                mhs_name = df_mahasiswa.loc[df_mahasiswa["id_mahasiswa"] == mhs_id, "nama_mahasiswa"].iloc[0]
                idx_top = np.argsort(hybrid_scores[i])[::-1][:top_n]
                
                df_tmp = df_komunitas.iloc[idx_top][["nama_komunitas", "id_komunitas", "kategori"]].copy()
                df_tmp["Skor_Hybrid"] = hybrid_scores[i][idx_top]
                df_tmp["Mahasiswa"] = mhs_name
                
                # Add detailed scores if requested
                if show_details:
                    df_tmp["CBF_Score"] = cbf_scores_norm[i][idx_top]
                    df_tmp["CF_Score"] = cf_scores_norm[i][idx_top]
                
                rekos.append(df_tmp)
        
        # Display results if we have recommendations
        if rekos:
            df_rekomendasi = pd.concat(rekos, ignore_index=True)
            
            # Display the recommendation table
            st.dataframe(df_rekomendasi.style.format({
                "Skor_Hybrid": "{:.4f}",
                "CBF_Score": "{:.4f}" if show_details else None,
                "CF_Score": "{:.4f}" if show_details else None
            }))
            
            # Download button
            st.download_button(
                "📥 Download Rekomendasi Hybrid", 
                df_rekomendasi.to_csv(index=False), 
                "rekomendasi_hybrid.csv",
                help="Download rekomendasi dalam format CSV"
            )
            
            # Prepare recommendations dictionary for the detailed display section
            recommendations = {}
            for i, mhs in df_mahasiswa.iterrows():
                mhs_id = mhs["id_mahasiswa"]
                idx_top = np.argsort(hybrid_scores[i])[::-1][:5]
                recommendations[mhs_id] = df_komunitas.iloc[idx_top]["id_komunitas"].tolist()
            
            # === 6. METHOD COMPARISON ===
            st.subheader("6️⃣ Perbandingan Metode Rekomendasi")
            
            # Pilih user untuk visualisasi
            sample_user_idx = 0
            if len(df_mahasiswa) > 1:
                sample_user = st.selectbox(
                    "Pilih mahasiswa untuk visualisasi perbandingan:",
                    df_mahasiswa["nama_mahasiswa"].tolist(),
                    index=0
                )
                sample_user_idx = df_mahasiswa[df_mahasiswa["nama_mahasiswa"] == sample_user].index[0]
            
            # Generate comparison visualization
            comparison_fig = plot_method_comparison(df_komunitas, cbf_scores_norm, cf_scores_norm, 
                                                   user_idx=sample_user_idx, top_n=5)
            st.pyplot(comparison_fig)
            
            # Analyze overlap between recommendations
            overlap_analysis = analyze_recommendation_overlap(df_komunitas, cbf_scores_norm, cf_scores_norm, 
                                                            sample_user_idx, top_n=5)
            
            with st.expander("Analisis Overlap Rekomendasi"):
                st.write("**Hasil Analisis Overlap:**")
                st.write(f"- Komunitas yang direkomendasikan oleh CBF dan CF: {overlap_analysis['cbf_cf_overlap']}")
                st.write(f"- Komunitas yang direkomendasikan oleh Simple Average yang juga direkomendasikan CBF: {overlap_analysis['avg_cbf_overlap']}")
                st.write(f"- Komunitas yang direkomendasikan oleh Simple Average yang juga direkomendasikan CF: {overlap_analysis['avg_cf_overlap']}")
                
                st.write("**Interpretasi:**")
                st.write(overlap_analysis['analysis'])
                st.write(overlap_analysis['dominance'])
            
            # === 7. DETAILED RECOMMENDATION DISPLAY ===
            st.header("📋 Detail Rekomendasi Per Mahasiswa")
            
            # Select a student to show recommendations for
            selected_user = st.selectbox(
                "Pilih Mahasiswa:", 
                df_mahasiswa['id_mahasiswa'].tolist(),
                format_func=lambda x: df_mahasiswa[df_mahasiswa['id_mahasiswa'] == x]['nama_mahasiswa'].iloc[0]
            )
            
            nama_selected = df_mahasiswa[df_mahasiswa['id_mahasiswa'] == selected_user]['nama_mahasiswa'].values[0]
            st.subheader(f"Rekomendasi untuk {nama_selected}")
            
            # Display student profile
            selected_profile = df_mahasiswa[df_mahasiswa['id_mahasiswa'] == selected_user]['profil_mahasiswa'].values[0]
            st.write("**Profil Mahasiswa:**", selected_profile)
            
            # Display student ratings if available
            selected_idx = df_mahasiswa[df_mahasiswa['id_mahasiswa'] == selected_user].index[0]
            
            if original_rating_cols:
                with st.expander("Rating yang Diberikan"):
                    user_ratings = df_mahasiswa.iloc[selected_idx][original_rating_cols]
                    
                    # Filter to show only non-zero ratings
                    non_zero_ratings = user_ratings[user_ratings > 0]
                    
                    if len(non_zero_ratings) > 0:
                        rating_df = pd.DataFrame({
                            'Komunitas': [col.replace('Rating', '').replace('[', '').replace(']', '').strip() 
                                        for col in non_zero_ratings.index],
                            'Rating': non_zero_ratings.values
                        })
                        
                        # Tampilkan dalam bentuk tabel
                        st.dataframe(rating_df)
                        
                        # Tambahkan visualisasi bar chart untuk rating
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(
                            x='Komunitas', 
                            y='Rating', 
                            data=rating_df, 
                            ax=ax, 
                            palette='viridis'
                        )
                        ax.set_title(f'Rating dari {nama_selected}')
                        ax.set_ylim(0, 5)
                        plt.xticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Tidak ada rating yang diberikan.")
            
            # Display recommendations for selected user
            st.write("**Top Rekomendasi:**")
            
            for idx, komunitas_id in enumerate(recommendations[selected_user]):
                if komunitas_id in df_komunitas['id_komunitas'].values:
                    komunitas_info = df_komunitas[df_komunitas['id_komunitas'] == komunitas_id].iloc[0]
                    
                    # Find this recommendation's scores
                    i = df_mahasiswa[df_mahasiswa['id_mahasiswa'] == selected_user].index[0]
                    kom_idx = df_komunitas[df_komunitas['id_komunitas'] == komunitas_id].index[0]
                    
                    hybrid_score = hybrid_scores[i][kom_idx]
                    cbf_score = cbf_scores_norm[i][kom_idx]
                    cf_score = cf_scores_norm[i][kom_idx]
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### {idx+1}. {komunitas_info['nama_komunitas']}")
                            st.caption(f"Kategori: {komunitas_info['kategori']}")
                            if 'deskripsi' in komunitas_info and komunitas_info['deskripsi']:
                                st.write(komunitas_info['deskripsi'])
                        
                        with col2:
                            st.metric(label="Skor Hybrid", value=f"{hybrid_score:.4f}")
                            
                            if show_details:
                                col2a, col2b = st.columns(2)
                                with col2a:
                                    st.metric(label="CBF", value=f"{cbf_score:.4f}")
                                with col2b:
                                    st.metric(label="CF", value=f"{cf_score:.4f}")
                        
                        # Show additional information if available
                        if ('aktivitas' in komunitas_info and komunitas_info['aktivitas']) or \
                           ('teknologi' in komunitas_info and komunitas_info['teknologi']) or \
                           ('visi_misi' in komunitas_info and komunitas_info['visi_misi']):
                            with st.expander("Detail Komunitas"):
                                cols = st.columns(3)
                                if 'aktivitas' in komunitas_info and komunitas_info['aktivitas']:
                                    with cols[0]:
                                        st.markdown("**Aktivitas:**")
                                        st.write(komunitas_info['aktivitas'])
                                
                                if 'teknologi' in komunitas_info and komunitas_info['teknologi']:
                                    with cols[1]:
                                        st.markdown("**Teknologi:**")
                                        st.write(komunitas_info['teknologi'])
                                
                                if 'visi_misi' in komunitas_info and komunitas_info['visi_misi']:
                                    with cols[2]:
                                        st.markdown("**Visi Misi:**")
                                        st.write(komunitas_info['visi_misi'])
                        
                        st.divider()
        else:
            st.warning("Tidak ada rekomendasi yang dihasilkan. Periksa data input.")