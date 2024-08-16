import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Sidebar for page navigation
page = st.sidebar.selectbox("Pilih Halaman", [
    "Jumlah Penduduk Miskin",
    "Indeks Kedalaman dan Keparahan Kemiskinan",
    "Garis Kemiskinan per Kabupaten/Kota pada Tahun Terpilih",
    # "Garis Kemiskinan per Kabupaten pada Tahun Prediksi"
])

# Page: Jumlah Penduduk Miskin
if page == "Jumlah Penduduk Miskin":
    chart_option = st.sidebar.selectbox(
        "Pilih Grafik yang Ingin Ditampilkan:",
        options=[
            "Jumlah Penduduk Miskin Aceh Tahun (2012-2021)",
            "Rata-rata Persentase Penduduk Miskin Menurut Daerah di Provinsi Aceh (2001-2022)"
        ]
    )

    if chart_option == "Jumlah Penduduk Miskin Aceh Tahun (2012-2021)":
        file_path1 = 'Jumlah Penduduk Miskin Provinsi Aceh Menurut KabupatenKota/merged_jumlah_penduduk_miskin_aceh.csv'
        data1 = pd.read_csv(file_path1)

        st.write("### Jumlah Penduduk Miskin Aceh Tahun (2012-2021)")
        st.markdown("""
        <p style="text-align: justify; text-indent: 30px;">
        Selama periode 2012 hingga 2021, terjadi fluktuasi dalam jumlah penduduk miskin di Aceh. 
        Jumlah penduduk miskin tertinggi tercatat pada tahun 2012, yaitu sekitar 880,52 ribu jiwa, 
        kemudian menurun secara bertahap hingga mencapai titik terendah pada tahun 2020 dengan jumlah 
        sekitar 814.93 ribu jiwa. Namun, pada tahun 2021, terjadi peningkatan kembali menjadi sekitar 834,25 ribu jiwa.
        </p>
        
        <p style="text-align: justify; text-indent: 30px;">
        Penurunan yang konsisten dari tahun 2013 hingga 2020 menunjukkan adanya perbaikan ekonomi atau 
        efektivitas program pengentasan kemiskinan di Aceh selama periode tersebut. Namun, peningkatan pada tahun 2021 
        mungkin terkait dengan faktor-faktor tertentu seperti pandemi COVID-19 atau kondisi ekonomi yang memburuk.
        </p>
        
        <p style="text-align: justify; text-indent: 30px;">
        Visualisasi berikut akan menampilkan tren jumlah penduduk miskin dari tahun 2012 hingga 2021. 
        Tren ini akan menunjukkan perubahan jumlah penduduk miskin setiap tahunnya.
        </p>
        """, unsafe_allow_html=True)

        # Data preparation for historical data
        data_grouped = data1.groupby('tahun').agg(
            bps_jumlah_penduduk=('bps_jumlah_penduduk', 'sum'),
            persentase_jumlah_penduduk_miskin=('persentase_jumlah_penduduk_miskin', 'mean')
        ).reset_index()

        # Model training for predictions
        model_penduduk = LinearRegression()
        X = data_grouped[['tahun']]
        y_penduduk = data_grouped['bps_jumlah_penduduk']
        model_penduduk.fit(X, y_penduduk)

        model_persentase = LinearRegression()
        y_persentase = data_grouped['persentase_jumlah_penduduk_miskin']
        model_persentase.fit(X, y_persentase)

        # Prediction for future years
        tahun_prediksi = pd.DataFrame({'tahun': [2022, 2023, 2024, 2025, 2026]})
        prediksi_jumlah_penduduk = model_penduduk.predict(tahun_prediksi)
        prediksi_persentase_miskin = model_persentase.predict(tahun_prediksi)

        prediksi_df = pd.DataFrame({
            'tahun': tahun_prediksi['tahun'],
            'bps_jumlah_penduduk': prediksi_jumlah_penduduk,
            'persentase_jumlah_penduduk_miskin': prediksi_persentase_miskin
        })

        # Combine historical and prediction data, adding a column to indicate the data type
        data_grouped['type'] = 'Actual'
        prediksi_df['type'] = 'Predicted'
        combined_df = pd.concat([data_grouped, prediksi_df], ignore_index=True)

        # Visualization for both historical and prediction data with color distinction
        fig = px.line(combined_df, 
                    x='tahun', 
                    y='bps_jumlah_penduduk', 
                    color='type',
                    title='Grafik Jumlah Penduduk Miskin dan Prediksi di Aceh (Hingga 2026)',
                    labels={'bps_jumlah_penduduk': 'Jumlah Penduduk (Ribu Jiwa)', 'tahun': 'Tahun', 'type': 'Data Type'},
                    height=500,
                    hover_data={'persentase_jumlah_penduduk_miskin': ':.2f'})

        # Update traces to differentiate colors between actual and predicted data
        fig.update_traces(mode='lines+markers', marker=dict(size=8))

        # Add a vertical line to separate actual and predicted data at 2021.5
        fig.add_vline(x=2021.5, line_width=2, line_dash='dash', line_color='yellow')

        fig.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        xaxis_title='Tahun',
                        yaxis_title='Jumlah Penduduk (Ribu Jiwa)',
                        xaxis=dict(tickformat='.0f'))

        st.plotly_chart(fig)

        st.write("### Jumlah Penduduk Miskin per Kab/Kota Tahun (2012-2021)")
        top_n_option = st.selectbox(
            "Pilih Jumlah Kabupaten/Kota Teratas",
            options=["3 Teratas", "5 Teratas", "10 Teratas", "Semua"]
        )

        if top_n_option == "3 Teratas":
            top_n = 3
        elif top_n_option == "5 Teratas":
            top_n = 5
        elif top_n_option == "10 Teratas":
            top_n = 10
        else:
            top_n = len(data1['bps_nama_kabupaten_kota'].unique())

        kab_kota_totals = data1.groupby('bps_nama_kabupaten_kota')['bps_jumlah_penduduk'].sum().reset_index()
        top_kab_kota = kab_kota_totals.sort_values(by='bps_jumlah_penduduk', ascending=False).head(top_n)['bps_nama_kabupaten_kota']
        filtered_data = data1[data1['bps_nama_kabupaten_kota'].isin(top_kab_kota)]

        fig2 = px.line(filtered_data, 
                       x='tahun', 
                       y='bps_jumlah_penduduk', 
                       color='bps_nama_kabupaten_kota',
                       title=f'Grafik Jumlah Penduduk Miskin per Kab/Kota Berdasarkan Tahun ({top_n_option})',
                       labels={'bps_jumlah_penduduk': 'Jumlah Penduduk (Ribu Jiwa)', 'tahun': 'Tahun', 'bps_nama_kabupaten_kota': 'Kabupaten/Kota'},
                       height=600)

        fig2.update_traces(mode='lines+markers', marker=dict(size=6), line=dict(width=1))
        fig2.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                           xaxis_title='Tahun',
                           yaxis_title='Jumlah Penduduk (Ribu Jiwa)')
        st.plotly_chart(fig2)

        st.write("### Prediksi Jumlah Penduduk Miskin dan Persentase di Aceh (2022-2026)")
        model_penduduk = LinearRegression()
        X = data_grouped[['tahun']]
        y_penduduk = data_grouped['bps_jumlah_penduduk']
        model_penduduk.fit(X, y_penduduk)

        model_persentase = LinearRegression()
        y_persentase = data_grouped['persentase_jumlah_penduduk_miskin']
        model_persentase.fit(X, y_persentase)

        tahun_prediksi = pd.DataFrame({'tahun': [2022, 2023, 2024, 2025, 2026]})
        prediksi_jumlah_penduduk = model_penduduk.predict(tahun_prediksi)
        prediksi_persentase_miskin = model_persentase.predict(tahun_prediksi)

        prediksi_df = pd.DataFrame({
            'tahun': tahun_prediksi['tahun'],
            'bps_jumlah_penduduk': prediksi_jumlah_penduduk,
            'persentase_jumlah_penduduk_miskin': prediksi_persentase_miskin
        })

        fig6 = px.line(prediksi_df, 
                    x='tahun', 
                    y='bps_jumlah_penduduk', 
                    title='Prediksi Jumlah Penduduk Miskin dan Persentase di Aceh (2022-2026)',
                    labels={'bps_jumlah_penduduk': 'Jumlah Penduduk (Ribu Jiwa)', 'tahun': 'Tahun'},
                    height=500,
                    hover_data={'persentase_jumlah_penduduk_miskin': ':.2f'})

        fig6.update_traces(mode='lines+markers', marker=dict(size=8), line=dict(width=1))
        fig6.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        xaxis_title='Tahun',
                        yaxis_title='Jumlah Penduduk (Ribu Jiwa)',
                        xaxis=dict(
                            tickformat='.0f',
                            tickmode='array',
                            tickvals=[2022, 2023, 2024, 2025, 2026],
                            range=[2021.5, 2026.5]
                        ))

        st.plotly_chart(fig6)

        st.write("""
        <p style='text-indent: 30px; text-align: justify;'>
        Langkah utama yang digunakan untuk menganalisis dan memprediksi data penduduk dan persentase kemiskinan di Aceh dari tahun 2022 hingga 2026 adalah mengolah data dengan mengelompokkan berdasarkan tahun, kemudian menghitung total jumlah penduduk dan rata-rata persentase penduduk miskin per tahun. Selanjutnya, dua model regresi linier dibangun menggunakan tahun sebagai variabel independen; satu model untuk memprediksi jumlah penduduk dan satu lagi untuk memprediksi persentase kemiskinan. Setelah model dilatih dengan data historis, prediksi untuk tahun-tahun mendatang hingga 2026 dilakukan. Hasil prediksi menunjukkan tren perubahan jumlah penduduk miskin dan persentase kemiskinan, yang dapat digunakan untuk perencanaan dan evaluasi kebijakan pengentasan kemiskinan di Aceh.
        </p>
        """, unsafe_allow_html=True)
    
    elif chart_option == "Rata-rata Persentase Penduduk Miskin Menurut Daerah di Provinsi Aceh (2001-2022)":
        # Load the data for rata-rata persentase penduduk miskin
        file_path3 = 'Jumlah Penduduk Miskin Provinsi Aceh Menurut KabupatenKota/persentase-penduduk-miskin-menurut-daerah-di-provinsi-aceh.csv'
        data3 = pd.read_csv(file_path3, delimiter=';')

        st.write("### Rata-rata Persentase Penduduk Miskin Menurut Daerah di Provinsi Aceh (2001-2022)")

        # Aggregating data by region
        filtered_data3 = data3[(data3['tahun'] >= 2001) & (data3['tahun'] <= 2022)]
        aggregated_data = filtered_data3.groupby('daerah', as_index=False).agg({'persentase_penduduk_miskin': 'mean'})

        fig4 = px.pie(
            aggregated_data,
            values='persentase_penduduk_miskin',
            names='daerah',
            title='Rata-rata Persentase Penduduk Miskin Menurut Daerah di Provinsi Aceh (2001-2022)',
            labels={
                'persentase_penduduk_miskin': 'Rata-rata Persentase Penduduk Miskin',
                'daerah': 'Daerah'
            },
            hover_data={
                'persentase_penduduk_miskin': ':.2f',
            }
        )

        st.plotly_chart(fig4)

    # Narasi setelah grafik dengan indentasi dan justify
        st.markdown("""
        <p style="text-align: justify; text-indent: 30px;">
        Visualisasi ini menampilkan perubahan rata-rata persentase penduduk miskin di Provinsi Aceh dari tahun 2001 hingga 2022, dengan pembagian antara daerah perkotaan dan perdesaan. Data menunjukkan bahwa:
        </p>
        <ul style="text-align: justify; text-indent: 30px;">
            <li><b>Daerah Perdesaan:</b>
                <ul>
                    <li>Secara konsisten, persentase penduduk miskin di daerah perdesaan lebih tinggi dibandingkan dengan perkotaan selama periode 2001-2022.</li>
                    <li>Pie chart menunjukkan bahwa mayoritas penduduk miskin di Provinsi Aceh berada di daerah perdesaan. Hal ini tercermin dari ukuran segmen yang lebih besar, menandakan persentase yang lebih tinggi.</li>
                </ul>
            </li>
            <li><b>Daerah Perkotaan:</b>
                <ul>
                    <li>Persentase penduduk miskin di perkotaan juga menunjukkan penurunan, pada tingkat yang lebih rendah dibandingkan perdesaan.</li>
                    <li>Meskipun lebih kecil, segmen perkotaan juga memiliki kontribusi signifikan dalam jumlah penduduk miskin, tetapi tetap lebih rendah dibandingkan dengan perdesaan.</li>
                </ul>
            </li>
        </ul>
        <p style="text-align: justify; text-indent: 30px;">
        Data ini menunjukkan adanya disparitas yang cukup signifikan antara daerah perkotaan dan perdesaan dalam hal kemiskinan. Meskipun terjadi penurunan secara keseluruhan di kedua daerah, daerah perdesaan cenderung memiliki persentase kemiskinan yang lebih tinggi sepanjang periode ini. Hal ini mungkin mencerminkan tantangan ekonomi yang lebih besar di daerah perdesaan dibandingkan dengan perkotaan.
        </p>
        """, unsafe_allow_html=True)
elif page == "Indeks Kedalaman dan Keparahan Kemiskinan":
    # Page: Indeks Kedalaman dan Keparahan Kemiskinan
    # Load the data for indeks kedalaman dan keparahan kemiskinan
        file_path2 = 'Indeks Kedalaman dan Keparahan Kemiskinan Aceh/test keparahan dan kedalaman.csv'
        data2 = pd.read_csv(file_path2)

        # Convert columns to appropriate data types
        data2['indeks_kedalaman'] = pd.to_numeric(data2['indeks_kedalaman'], errors='coerce')
        data2['indeks_keparahan_kemiskinan'] = pd.to_numeric(data2['indeks_keparahan_kemiskinan'], errors='coerce')
        data2['tahun'] = pd.to_numeric(data2['tahun'], errors='coerce')

        # Group by year and calculate the average for both indices
        avg_data = data2.groupby('tahun').agg({
            'indeks_kedalaman': 'mean',
            'indeks_keparahan_kemiskinan': 'mean'
        }).reset_index()

        # Calculate percentage change from the previous year
        avg_data['perc_change_kedalaman'] = avg_data['indeks_kedalaman'].pct_change() * 100
        avg_data['perc_change_keparahan'] = avg_data['indeks_keparahan_kemiskinan'].pct_change() * 100

        # Fill NaN values resulting from pct_change calculation with 0
        avg_data.fillna(0, inplace=True)

        # Train linear regression models for both indices
        X = avg_data['tahun'].values.reshape(-1, 1)
        model_depth = LinearRegression().fit(X, avg_data['indeks_kedalaman'])
        model_severity = LinearRegression().fit(X, avg_data['indeks_keparahan_kemiskinan'])

        # Predict future years (2024-2028)
        future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)
        predicted_depth = model_depth.predict(future_years)
        predicted_severity = model_severity.predict(future_years)

        # Combine the predictions with future years for visualization
        future_data = pd.DataFrame({
            'tahun': future_years.flatten(),
            'indeks_kedalaman': predicted_depth,
            'indeks_keparahan_kemiskinan': predicted_severity
        })

        # Combine past and predicted data for plotting
        combined_data = pd.concat([avg_data, future_data])

        # Create a figure for both historical data and predictions
        fig = go.Figure()

        # Add poverty depth index line (historical)
        fig.add_trace(go.Scatter(
            x=avg_data['tahun'], y=avg_data['indeks_kedalaman'],
            mode='lines+markers+text',
            name='Indeks Kedalaman (2005-2023)',
            text=avg_data['indeks_kedalaman'].round(2),
            textposition='top center',
            hovertemplate=(
                'Tahun: %{x}<br>'
                'Indeks Kedalaman: %{y:.2f}<br>'
                'Indeks Keparahan: %{customdata[0]:.2f}<br>'
                '<br>'
                'Peningkatan Kedalaman: %{customdata[1]:.2f}%<br>'
                '<extra></extra>'
            ),
            customdata=avg_data[['indeks_keparahan_kemiskinan', 'perc_change_kedalaman']].values
        ))

        # Add poverty severity index line (historical)
        fig.add_trace(go.Scatter(
            x=avg_data['tahun'], y=avg_data['indeks_keparahan_kemiskinan'],
            mode='lines+markers+text',
            name='Indeks Keparahan (2005-2023)',
            text=avg_data['indeks_keparahan_kemiskinan'].round(2),
            textposition='top center',
            hovertemplate=(
                'Tahun: %{x}<br>'
                'Indeks Keparahan: %{y:.2f}<br>'
                'Indeks Kedalaman: %{customdata[1]:.2f}<br>'
                '<br>'
                'Peningkatan Keparahan: %{customdata[0]:.2f}%<br>'
                '<extra></extra>'
            ),
            customdata=avg_data[['indeks_keparahan_kemiskinan', 'perc_change_keparahan']].values
        ))

        # Add poverty depth index line (predictions)
        fig.add_trace(go.Scatter(
            x=future_data['tahun'], y=future_data['indeks_kedalaman'],
            mode='lines+markers+text',
            name='Indeks Kedalaman (Prediksi 2024-2028)',
            text=future_data['indeks_kedalaman'].round(2),
            textposition='top center',
            hovertemplate=(
                'Tahun: %{x}<br>'
                'Indeks Kedalaman: %{y:.2f}<br>'
                'Indeks Keparahan: %{customdata[0]:.2f}<br>'
                '<extra></extra>'
            ),
            customdata=np.stack((future_data['indeks_keparahan_kemiskinan'], future_data['indeks_kedalaman']), axis=-1)
        ))

        # Add poverty severity index line (predictions)
        fig.add_trace(go.Scatter(
            x=future_data['tahun'], y=future_data['indeks_keparahan_kemiskinan'],
            mode='lines+markers+text',
            name='Indeks Keparahan (Prediksi 2024-2028)',
            text=future_data['indeks_keparahan_kemiskinan'].round(2),
            textposition='top center',
            hovertemplate=(
                'Tahun: %{x}<br>'
                'Indeks Keparahan: %{y:.2f}<br>'
                'Indeks Kedalaman: %{customdata[1]:.2f}<br>'
                '<extra></extra>'
            ),
            customdata=np.stack((future_data['indeks_keparahan_kemiskinan'], future_data['indeks_kedalaman']), axis=-1)
        ))

        # Add a vertical line at the position 2023.5 to separate historical and predicted data
        fig.add_shape(
            dict(
                type="line",
                x0=2023.5,
                y0=0,
                x1=2023.5,
                y1=max(combined_data['indeks_kedalaman'].max(), combined_data['indeks_keparahan_kemiskinan'].max()),
                line=dict(color="Yellow", width=2, dash="dash"),
            )
        )

        # Update layout to include titles and axis labels, with increased width
        fig.update_layout(
            title='Grafik Indeks Kedalaman dan Keparahan Kemiskinan (2005-2028)',
            xaxis_title='Tahun',
            yaxis_title='Nilai Indeks',
            legend_title_text='Indeks',
            height=500,
            width=1800  
        )

        st.write("### Indeks Kedalaman dan Keparahan Kemiskinan (2005-2028)")
        st.plotly_chart(fig)


        st.write("""
        <p style='text-indent: 30px; text-align: justify;'>
        Visualisasi ini menampilkan perkembangan Indeks Kedalaman dan Keparahan Kemiskinan di Indonesia selama periode 2005 hingga 2023,
        yang diwakili oleh total indeks yang dihitung dari penjumlahan antara indeks kedalaman dan keparahan kemiskinan setiap tahunnya. 
        Secara umum, grafik menunjukkan adanya fluktuasi yang signifikan sepanjang periode ini. Pada tahun 2005, indeks total berada pada 
        angka 5.98, mencerminkan tingkat kemiskinan yang cukup dalam dan parah di berbagai wilayah pada tahun tersebut. 
        Setelah itu, terjadi penurunan yang cukup konsisten hingga tahun 2009, yang kemungkinan mencerminkan keberhasilan kebijakan penanggulangan 
        kemiskinan atau perbaikan kondisi ekonomi pada masa itu. 
        </p>
        
        <p style='text-indent: 30px; text-align: justify;'>
        Namun, setelah periode tersebut, terlihat adanya fluktuasi dalam indeks total, 
        dengan beberapa tahun mencatat peningkatan yang mungkin disebabkan oleh kondisi sosial-ekonomi yang menantang atau perubahan dalam kebijakan
        pemerintah. Kenaikan dan penurunan indeks ini mencerminkan dinamika kompleks dari kemiskinan di Indonesia, yang dipengaruhi oleh berbagai faktor 
        seperti pertumbuhan ekonomi, kebijakan sosial, serta kejadian-kejadian global yang berdampak pada kesejahteraan masyarakat. 
        Dengan memahami pola ini, para pembuat kebijakan dan pemangku kepentingan lainnya dapat lebih tepat dalam merumuskan strategi yang efektif untuk 
        mengatasi kemiskinan di masa depan.
        </p>
        """, unsafe_allow_html=True)

        st.write(f"### Indeks Kedalaman dan Keparahan Kemiskinan per Kabupaten/Kota")

        # Select the number of top regions to display
        top_n = st.selectbox("Pilih jumlah Kota/Kabupaten teratas:", [3, 5, 10], index=0)

        # Calculate mean indices per region
        region_data_mean = data2.groupby('bps_nama_kabupaten_kota').agg({
            'indeks_kedalaman': 'mean',
            'indeks_keparahan_kemiskinan': 'mean'
        }).reset_index()

        # Sort regions by poverty severity and get the top N regions
        top_regions = region_data_mean.nlargest(top_n, 'indeks_keparahan_kemiskinan')['bps_nama_kabupaten_kota']

        # Filter data for top N regions
        filtered_data = data2[data2['bps_nama_kabupaten_kota'].isin(top_regions)]

        fig7 = go.Figure()

        # Loop through each of the top N regions to create a separate line for each
        for region in filtered_data['bps_nama_kabupaten_kota'].unique():
            region_data = filtered_data[filtered_data['bps_nama_kabupaten_kota'] == region]

            # Add poverty depth index line for each region
            fig7.add_trace(go.Scatter(
                x=region_data['tahun'],
                y=region_data['indeks_kedalaman'],
                mode='lines+markers',
                name=f'{region}',
                hovertemplate=(
                    f'Tahun: %{{x}}<br>'
                    f'Kabupaten/Kota: {region}<br>'
                    f'Indeks Kedalaman: %{{y:.2f}}<br>'
                    f'Indeks Keparahan: %{{customdata:.2f}}<br>'
                    '<extra></extra>'
                ),
                customdata=region_data['indeks_keparahan_kemiskinan'].values  # Hover data only
            ))

        # Update layout to include titles and axis labels with larger size
        fig7.update_layout(
            title=f'Grafik Indeks Kedalaman dan Keparahan Kemiskinan (Top {top_n})',
            xaxis_title='Tahun',
            yaxis_title='Nilai Indeks',
            legend_title_text='Kabupaten/Kota',
            height=600,
            width=1500
        )

        # Show the new chart
        st.plotly_chart(fig7)

        # Train linear regression models for both indices
        X = avg_data['tahun'].values.reshape(-1, 1)
        model_depth = LinearRegression().fit(X, avg_data['indeks_kedalaman'])
        model_severity = LinearRegression().fit(X, avg_data['indeks_keparahan_kemiskinan'])

        # Predict future years (2024-2028)
        future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)
        predicted_depth = model_depth.predict(future_years)
        predicted_severity = model_severity.predict(future_years)

        # Combine the predictions with future years for visualization
        future_data = pd.DataFrame({
            'tahun': future_years.flatten(),
            'indeks_kedalaman': predicted_depth,
            'indeks_keparahan_kemiskinan': predicted_severity
        })

        # Create a figure for the predictions
        fig_pred = go.Figure()

        # Add poverty depth index line (only for predicted years)
        fig_pred.add_trace(go.Scatter(
            x=future_data['tahun'], y=future_data['indeks_kedalaman'],
            mode='lines+markers+text',
            name='Indeks Kedalaman (Prediksi)',  
            text=future_data['indeks_kedalaman'].round(2),
            textposition='top center',
            hovertemplate=(
                'Tahun: %{x}<br>'
                'Indeks Kedalaman: %{y:.2f}<br>'
                'Indeks Keparahan: %{customdata[0]:.2f}<br>'
                '<extra></extra>'
            ),
            customdata=np.stack((future_data['indeks_keparahan_kemiskinan'], future_data['indeks_kedalaman']), axis=-1)
        ))

        # Add poverty severity index line (only for predicted years)
        fig_pred.add_trace(go.Scatter(
            x=future_data['tahun'], y=future_data['indeks_keparahan_kemiskinan'],
            mode='lines+markers+text',
            name='Indeks Keparahan (Prediksi)',  
            text=future_data['indeks_keparahan_kemiskinan'].round(2),
            textposition='top center',
            hovertemplate=(
                'Tahun: %{x}<br>'
                'Indeks Keparahan: %{y:.2f}<br>'
                'Indeks Kedalaman: %{customdata[1]:.2f}<br>'
                '<extra></extra>'
            ),
            customdata=np.stack((future_data['indeks_keparahan_kemiskinan'], future_data['indeks_kedalaman']), axis=-1)
        ))

        # Update layout to include titles and axis labels
        fig_pred.update_layout(
            title='Prediksi Indeks Kedalaman dan Keparahan Kemiskinan (2024-2028)',
            xaxis_title='Tahun',
            yaxis_title='Nilai Indeks',
            legend_title_text='Indeks'
        )

        st.write("### Prediksi Indeks Kedalaman dan Keparahan Kemiskinan (2024-2028)")
        st.plotly_chart(fig_pred)

        st.markdown("""
        <p style='text-indent: 30px; text-align: justify;'>
        Grafik diatas merupakan prediksi terhadap indeks kedalaman dan keparahan kemiskinan untuk tahun 2024 hingga 2028, dan menampilkan hasil prediksi tersebut dalam bentuk grafik interaktif. Pertama, data yang ada dikonversi menjadi tipe numerik untuk memastikan bahwa perhitungan dapat dilakukan tanpa kesalahan, khususnya pada kolom indeks_kedalaman, indeks_keparahan_kemiskinan, dan tahun. Data ini kemudian dikelompokkan berdasarkan tahun, dan rata-rata dari masing-masing indeks dihitung untuk setiap tahun, menghasilkan DataFrame baru yang berisi nilai rata-rata per tahun.
        </p>

        <p style='text-indent: 30px; text-align: justify;'>
        Selanjutnya, dua model regresi linier dibangun menggunakan tahun sebagai variabel independen. Model pertama digunakan untuk memprediksi indeks_kedalaman, sementara model kedua digunakan untuk memprediksi indeks_keparahan_kemiskinan. Dengan menggunakan model ini, prediksi dilakukan untuk tahun-tahun mendatang (2024-2028), dan hasil prediksi tersebut disusun dalam sebuah DataFrame baru.
        </p>
        """, unsafe_allow_html=True)

# Page: Garis Kemiskinan per Kabupaten/Kota pada Tahun Terpilih
elif page == "Garis Kemiskinan per Kabupaten/Kota pada Tahun Terpilih":
        st.write("### Garis Kemiskinan per Kabupaten/Kota Tahun (2010-2023)")

        # Load data for selected year
        file_path4 = 'Garis Kemiskinan (GK) Aceh/garis_kemiskinan_rupiah.csv'
        data2 = pd.read_csv(file_path4, delimiter=';')

        # st.write("Available columns:", data2.columns.tolist())  # Print column names

        tahun_terpilih = st.selectbox("Pilih Tahun", options=data2['tahun'].unique())

        data_year = data2[data2['tahun'] == tahun_terpilih]

        fig3 = px.bar(data_year,
                    x='bps_nama_kabupaten_kota',
                    y='garis_kemiskinan',
                    title=f'Garis Kemiskinan per Kabupaten/Kota pada Tahun {tahun_terpilih}',
                    labels={'bps_nama_kabupaten_kota': 'Kabupaten/Kota', 'garis_kemiskinan': 'Garis Kemiskinan'},
                    height=600)

        fig3.update_layout(xaxis_title='Kabupaten/Kota',
                        yaxis_title='Garis Kemiskinan',
                        xaxis_tickangle=-45)
        st.plotly_chart(fig3)

        st.write("""
        <p style='text-indent: 20px; text-align: justify;'>
        Garis kemiskinan menggambarkan batas minimum pendapatan atau konsumsi yang diperlukan untuk memenuhi kebutuhan dasar di setiap kabupaten/kota. Pada tahun yang terpilih, variasi garis kemiskinan di Aceh dapat dilihat melalui visualisasi ini. Data ini penting untuk mengidentifikasi wilayah yang membutuhkan perhatian khusus dalam program pengentasan kemiskinan.
        </p>
        """, unsafe_allow_html=True)

        st.write("### Prediksi Garis Kemiskinan per Kabupaten/Kota Tahun (2024-2028)")
        # Multiselect menu for selecting multiple kabupaten/kota
        options = [
            'Kabupaten Simeulue', 'Kabupaten Aceh Singkil', 'Kabupaten Aceh Selatan', 
            'Kabupaten Aceh Tenggara', 'Kabupaten Aceh Timur', 'Kabupaten Aceh Tengah', 
            'Kabupaten Aceh Barat', 'Kabupaten Aceh Besar', 'Kabupaten Pidie', 
            'Kabupaten Bireuen', 'Kabupaten Aceh Utara', 'Kabupaten Aceh Barat Daya', 
            'Kabupaten Gayo Lues', 'Kabupaten Aceh Tamiang', 'Kabupaten Nagan Raya', 
            'Kabupaten Aceh Jaya', 'Kabupaten Bener Meriah', 'Kabupaten Pidie Jaya', 
            'Kota Banda Aceh', 'Kota Sabang', 'Kota Langsa', 
            'Kota Lhokseumawe', 'Kota Subulussalam'
        ]

        # Add a "Select All" option at the beginning of the list
        options = ["Select All"] + options

        # Multiselect menu for selecting multiple kabupaten/kota
        selected_kabupatens = st.multiselect('Pilih Kabupaten/Kota:', options)

        # If "Select All" is chosen, display all kabupaten/kota
        if "Select All" in selected_kabupatens:
            selected_kabupatens = options[1:]  # Exclude the "Select All" option itself

        # Linear regression model to predict future poverty lines
        future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)

        # Creating an interactive line chart
        fig = go.Figure()

        # Loop through selected kabupaten/kota and add a line for each
        for selected_kabupaten in selected_kabupatens:
            subset = data2[data2['bps_nama_kabupaten_kota'] == selected_kabupaten]

            # Check if there is data available for the selected kabupaten/kota
            if not subset.empty:
                X = subset['tahun'].values.reshape(-1, 1)
                y = subset['garis_kemiskinan'].values

                if len(X) > 0:  # Ensure there are samples to train on
                    model = LinearRegression().fit(X, y)
                    pred = model.predict(future_years)

                    # Creating DataFrame for prediction results
                    pred_df = pd.DataFrame({
                        'tahun': future_years.flatten(),
                        'garis_kemiskinan': pred
                    })

                    # Adding trace for each selected kabupaten/kota
                    fig.add_trace(go.Scatter(
                        x=pred_df['tahun'],
                        y=pred_df['garis_kemiskinan'],
                        mode='lines+markers',
                        name=selected_kabupaten,
                        hovertemplate="<b>%{fullData.name}</b><br>Tahun: %{x}<br>Garis Kemiskinan: Rp%{y:,.0f}<extra></extra>"
                    ))
            else:
                st.warning(f"Tidak ada data untuk {selected_kabupaten}")

        # Enhancing the visualization
        fig.update_layout(
            width=1200,
            height=600,
            title="Prediksi Garis Kemiskinan per Kabupaten/Kota Tahun (2024-2028)",
            xaxis_title='Tahun',
            yaxis_title='Garis Kemiskinan (Rupiah)',
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            legend_title_text='Kabupaten/Kota'
        )

        # Display the plot in Streamlit
        if selected_kabupatens:
            st.plotly_chart(fig)

            st.write("""
            <p style='text-indent: 20px; text-align: justify;'>
            Untuk memprediksi dan memahami perubahan garis kemiskinan di setiap kabupaten/kota di Aceh dalam lima tahun mendatang (2024-2028), proses dimulai dengan pengumpulan data historis mengenai garis kemiskinan dari tahun-tahun sebelumnya. Dengan data ini, model regresi linear dibangun untuk masing-masing kabupaten/kota. Regresi linear, sebagai teknik statistik, memungkinkan kita memprediksi nilai garis kemiskinan di masa depan berdasarkan tren historis. Setelah model dilatih, prediksi nilai garis kemiskinan untuk tahun-tahun yang akan datang dihasilkan. Hasil prediksi ini disimpan dalam dictionary yang kemudian diubah menjadi DataFrame untuk memudahkan analisis lebih lanjut. DataFrame ini memungkinkan pembuatan visualisasi seperti grafik garis waktu yang menunjukkan perubahan garis kemiskinan dari tahun ke tahun dan peta tematik yang menggambarkan prediksi garis kemiskinan untuk setiap kabupaten/kota. Visualisasi ini membantu pembuat kebijakan dalam mengidentifikasi daerah yang mungkin memerlukan intervensi khusus dan merencanakan alokasi sumber daya yang lebih efisien, sehingga strategi pengentasan kemiskinan dapat disesuaikan dengan kebutuhan nyata di masing-masing wilayah.
            </p>
            """, unsafe_allow_html=True)
        else:
            st.warning("Silakan pilih setidaknya satu kabupaten/kota untuk melihat hasil prediksi.")