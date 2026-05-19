import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from numpy.random import randn

# Настройки графиков
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True


# ==========================================
# 1. ГЕНЕРАЦИЯ PSS (по 3GPP 36.211 - 6.11.1)
# ==========================================
def generate_pss(N_ID_2):
    """
    Генерация PSS для заданного N_ID_2 (0, 1, или 2).
    Возвращает 62 комплексных символа.
    """
    u_shift = [25, 29, 34]
    u = u_shift[N_ID_2]
    d_u = np.zeros(62, dtype=complex)

    for n in range(62):
        if n <= 30:
            d_u[n] = np.exp(-1j * np.pi * u * n * (n + 1) / 63)
        else:
            d_u[n] = np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63)
    return d_u


# ==========================================
# 2. ГЕНЕРАЦИЯ SSS (по 3GPP 36.211 - 6.11.2)
# ==========================================
def generate_sss(N_ID_1, N_ID_2):
    """
    Генерация SSS для подкадра 0.
    N_ID_1: 0...167, N_ID_2: 0...2
    Возвращает 62 символа.
    """
    q_prime = N_ID_1 // 30
    q = (N_ID_1 + q_prime * (q_prime + 1) // 2) // 30
    m_prime = N_ID_1 + q * (q + 1) // 2
    m0 = m_prime % 31
    m1 = (m0 + (m_prime // 31) + 1) % 31

    # Сдвиговые регистры для m-последовательностей
    def shift_reg(init, taps):
        x = np.array(init)
        for _ in range(26):
            x = np.append(x, np.sum(x[taps]) % 2)
        return 1 - 2 * x

    x_s = shift_reg([0, 0, 0, 0, 1], [2, 0])  # x_s(i+5) = x_s(i+2) + x_s(i)
    x_c = shift_reg([0, 0, 0, 0, 1], [3, 0])  # x_c(i+5) = x_c(i+3) + x_c(i)
    x_z = shift_reg([0, 0, 0, 0, 1], [4, 2, 1, 0])  # x_z(i+5) = x_z(i+4)+x_z(i+2)+x_z(i+1)+x_z(i)

    s_tilda = x_s
    c_tilda = x_c
    z_tilda = x_z

    # Четные поднесущие (even)
    s0_m0_even = s_tilda[(np.arange(31) + m0) % 31]
    s1_m1_even = s_tilda[(np.arange(31) + m1) % 31]
    c0_even = c_tilda[(np.arange(31) + N_ID_2) % 31]
    d_even_sub0 = s0_m0_even * c0_even

    # Нечетные поднесущие (odd)
    s1_m1_odd = s_tilda[(np.arange(31) + m1) % 31]
    c1_odd = c_tilda[(np.arange(31) + N_ID_2 + 3) % 31]
    z1_m0_odd = z_tilda[(np.arange(31) + (m0 % 8)) % 31]
    d_odd_sub0 = s1_m1_odd * c1_odd * z1_m0_odd

    # Чередование even/odd
    d_sub0 = np.zeros(62)
    d_sub0[::2] = d_even_sub0
    d_sub0[1::2] = d_odd_sub0
    return d_sub0


# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ КОРРЕЛЯЦИИ
# ==========================================
def cross_correlate(x, y):
    """Кросс-корреляция с использованием scipy (аналог dsp.Crosscorrelator)"""
    return correlate(x, y, mode='full')


def add_awgn(signal, snr_db):
    """Добавление аддитивного белого гауссовского шума"""
    E_avg = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(E_avg / (2 * snr_linear))
    noise = sigma * (randn(len(signal)) + 1j * randn(len(signal)))
    return signal + noise


# ==========================================
# 4. ПРИЕМНИК PSS/SSS (МОДЕЛИРОВАНИЕ)
# ==========================================
def pss_detector(received_signal, pss_candidates):
    """
    Приемник PSS: корреляция принятого сигнала с тремя кандидатами.
    Возвращает индекс лучшего совпадения и значение корреляции.
    """
    correlations = []
    for pss in pss_candidates:
        # Кросс-корреляция
        corr = cross_correlate(received_signal, np.conj(pss))
        # Берем центральную часть (задержка от 0 до 61)
        corr_centered = np.abs(corr[61:123])
        correlations.append(np.max(corr_centered))
    best_idx = np.argmax(correlations)
    return best_idx, correlations[best_idx]


def frequency_offset_estimation(received_pss, expected_pss, sample_rate=30.72e6):
    """Оценка частотного сдвига по фазе корреляции"""
    phase_diff = np.angle(np.dot(received_pss, np.conj(expected_pss)))
    time_duration = 62 / sample_rate
    freq_offset = (phase_diff / (2 * np.pi)) / time_duration
    return freq_offset


def correct_frequency_offset(signal, freq_offset, sample_rate=30.72e6):
    """Коррекция частотного сдвига для каждого отсчета"""
    time = np.arange(len(signal)) / sample_rate
    correction = np.exp(-1j * 2 * np.pi * freq_offset * time)
    return signal * correction


# ==========================================
# 5. ВИЗУАЛИЗАЦИЯ РЕСУРСНОЙ СЕТКИ И PSS/SSS
# ==========================================
def plot_resource_grid(pss, sss, bandwidth_rb=6):
    """
    Отображение ресурсной сетки для одного субфрейма.
    Показывает расположение PSS/SSS на центральных 6 RB.
    """
    subcarriers = 12 * bandwidth_rb  # 72 для 6 RB
    ofdm_symbols = 14  # Normal CP

    grid = np.zeros((subcarriers, ofdm_symbols), dtype=complex)

    # Размещаем SSS в символе 5 (последний символ слота 0 в FDD)
    sss_start = subcarriers // 2 - 31
    grid[sss_start - 1:sss_start + 61, 5] = sss  # DC-поднесущая пропускается

    # Размещаем PSS в символе 6 (как в примерах ShareTechnote)
    pss_start = subcarriers // 2 - 31
    grid[pss_start - 1:pss_start + 61, 6] = pss

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(grid), aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.title('Resource Grid (6 RB, Normal CP) – Subframe 0 (FDD)')
    plt.xlabel('OFDM Symbol Index')
    plt.ylabel('Subcarrier Index')

    # Метки для PSS и SSS
    plt.text(6, subcarriers // 2 + 35, 'PSS', ha='center', color='cyan', fontweight='bold')
    plt.text(5, subcarriers // 2 + 35, 'SSS', ha='center', color='lime', fontweight='bold')
    plt.axhline(y=subcarriers // 2 - 36, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=subcarriers // 2 + 35, color='black', linestyle='--', alpha=0.5)
    plt.ylim(subcarriers - 1, 0)
    plt.tight_layout()


# ==========================================
# ОСНОВНОЙ БЛОК МОДЕЛИРОВАНИЯ
# ==========================================
if __name__ == "__main__":
    # Параметры ячейки
    N_ID_2 = 0  # Идентификатор сектора (0,1,2)
    N_ID_1 = 10  # Идентификатор группы (0...167)
    SNR_dB = 20  # Отношение сигнал/шум

    # --- Генерация PSS и SSS ---
    pss_ideal = generate_pss(N_ID_2)
    sss_ideal = generate_sss(N_ID_1, N_ID_2)

    print(f"Сгенерирован PSS (N_ID_2={N_ID_2}) длиной {len(pss_ideal)}")
    print(f"Сгенерирован SSS (N_ID_1={N_ID_1}, N_ID_2={N_ID_2}) длиной {len(sss_ideal)}")

    # --- График 1: Созвездие PSS/SSS ---
    fig1, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(pss_ideal.real, pss_ideal.imag, 'ro', markersize=4)
    axes[0].set_title('PSS Constellation (Zadoff-Chu)')
    axes[0].set_xlabel('Real');
    axes[0].set_ylabel('Imag')
    axes[0].axis('equal');
    axes[0].grid(True)

    axes[1].plot(sss_ideal.real, sss_ideal.imag, 'bo', markersize=4)
    axes[1].set_title('SSS Constellation (m-sequence based)')
    axes[1].set_xlabel('Real');
    axes[1].set_ylabel('Imag')
    axes[1].axis('equal');
    axes[1].grid(True)
    plt.suptitle(f'Сгенерированные сигналы (N_ID_2={N_ID_2}, N_ID_1={N_ID_1})')
    plt.tight_layout()

    # --- График 2: Кросс-корреляция между разными PSS  ---
    pss0 = generate_pss(0)
    pss1 = generate_pss(1)
    pss2 = generate_pss(2)

    corr_0_0 = cross_correlate(pss0, np.conj(pss0))
    corr_0_1 = cross_correlate(pss0, np.conj(pss1))
    corr_0_2 = cross_correlate(pss0, np.conj(pss2))

    taps = np.arange(62)
    fig2, axes_corr = plt.subplots(3, 1, figsize=(12, 8))
    axes_corr[0].stem(taps, np.abs(corr_0_0[61:123]))
    axes_corr[0].set_title('Корреляция PSS(NID0) и PSS(NID0)')
    axes_corr[0].set_ylim(0, 70)

    axes_corr[1].stem(taps, np.abs(corr_0_1[61:123]))
    axes_corr[1].set_title('Корреляция PSS(NID0) и PSS(NID1)')
    axes_corr[1].set_ylim(0, 70)

    axes_corr[2].stem(taps, np.abs(corr_0_2[61:123]))
    axes_corr[2].set_title('Корреляция PSS(NID0) и PSS(NID2)')
    axes_corr[2].set_ylim(0, 70)
    plt.suptitle('Свойства кросс-корреляции PSS (низкая взаимная корреляция)')
    plt.tight_layout()

    # --- График 3: Устойчивость к фазовому сдвигу и шуму ---
    ph_shift = np.pi / 3
    pss_shifted = pss0 * np.exp(1j * ph_shift)
    corr_shifted = cross_correlate(pss0, np.conj(pss_shifted))

    fig3, axes_shift = plt.subplots(2, 2, figsize=(12, 8))
    axes_shift[0, 0].plot(pss0.real, pss0.imag, 'ro', markersize=4)
    axes_shift[0, 0].set_title('Оригинальный PSS')
    axes_shift[0, 0].axis('equal')

    axes_shift[0, 1].plot(pss_shifted.real, pss_shifted.imag, 'bo', markersize=4)
    axes_shift[0, 1].set_title(f'PSS с фазовым сдвигом {ph_shift:.2f} рад')
    axes_shift[0, 1].axis('equal')

    axes_shift[1, 0].stem(taps, np.abs(corr_shifted[61:123]))
    axes_shift[1, 0].set_title('Амплитуда корреляции (нечувствительна к сдвигу)')
    axes_shift[1, 0].set_ylim(0, 70)

    axes_shift[1, 1].stem(taps, np.angle(corr_shifted[61:123]))
    axes_shift[1, 1].set_title('Фаза корреляции (показывает величину сдвига)')
    axes_shift[1, 1].set_ylim(-np.pi, np.pi)
    plt.suptitle('Поведение при фазовом сдвиге')
    plt.tight_layout()

    # --- График 4: Шумовая устойчивость ---
    pss_noisy = add_awgn(pss0, SNR_dB)
    corr_noisy = cross_correlate(pss0, np.conj(pss_noisy))

    fig4, axes_noise = plt.subplots(2, 1, figsize=(12, 8))
    axes_noise[0].plot(pss0.real, pss0.imag, 'ro', label='Идеальный', markersize=4)
    axes_noise[0].plot(pss_noisy.real, pss_noisy.imag, 'k.', label=f'С шумом ({SNR_dB} дБ)', alpha=0.5)
    axes_noise[0].set_title(f'Созвездие PSS с AWGN (SNR = {SNR_dB} дБ)')
    axes_noise[0].axis('equal');
    axes_noise[0].legend()

    axes_noise[1].stem(taps, np.abs(corr_noisy[61:123]))
    axes_noise[1].set_title('Корреляция зашумленного PSS с идеальным')
    axes_noise[1].set_ylim(0, 70)
    plt.suptitle('Помехоустойчивость PSS')
    plt.tight_layout()

    # --- Моделирование приемника и частотной коррекции ---
    print("\n--- Моделирование приемника ---")

    # Имитация принятого сигнала PSS с частотным сдвигом 500 Гц
    sample_rate = 30.72e6
    freq_offset_true = 500  # Гц
    time_vector = np.arange(62) / sample_rate
    pss_received = pss0 * np.exp(1j * 2 * np.pi * freq_offset_true * time_vector)
    pss_received = add_awgn(pss_received, SNR_dB)  # добавляем шум

    # Обнаружение PSS
    candidates = [generate_pss(i) for i in range(3)]
    detected_id, corr_val = pss_detector(pss_received, candidates)
    print(f"Обнаружен N_ID_2 = {detected_id} (истинное значение {N_ID_2})")

    # Оценка и коррекция частотного сдвига
    estimated_offset = frequency_offset_estimation(pss_received, pss0)
    print(f"Истинный частотный сдвиг: {freq_offset_true} Гц")
    print(f"Оцененный частотный сдвиг: {estimated_offset:.1f} Гц")

    pss_corrected = correct_frequency_offset(pss_received, estimated_offset)

    # График частотной коррекции
    fig5, axes_freq = plt.subplots(1, 2, figsize=(12, 5))
    axes_freq[0].plot(pss_received.real[:30], label='До коррекции')
    axes_freq[0].plot(pss_corrected.real[:30], '--', label='После коррекции')
    axes_freq[0].set_title('Сравнение реальной части до/после частотной коррекции')
    axes_freq[0].legend()

    axes_freq[1].plot(np.angle(pss_received[:30]), label='Фаза до коррекции')
    axes_freq[1].plot(np.angle(pss_corrected[:30]), '--', label='Фаза после коррекции')
    axes_freq[1].set_title('Фаза сигнала до/после коррекции')
    axes_freq[1].legend()
    plt.tight_layout()

    # --- Ресурсная сетка с PSS и SSS ---
    plot_resource_grid(pss_ideal, sss_ideal, bandwidth_rb=6)

    # --- Сравнение всех трех PSS (как в таблице ShareTechnote) ---
    fig6, axes_all = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['red', 'green', 'blue']
    for i, (pss, col) in enumerate(zip(candidates, colors)):
        axes_all[i].plot(pss.real, pss.imag, 'o', color=col, markersize=4)
        axes_all[i].set_title(f'N_ID_2 = {i}')
        axes_all[i].set_xlabel('Real');
        axes_all[i].set_ylabel('Imag')
        axes_all[i].axis('equal');
        axes_all[i].grid(True)
    plt.suptitle('Три уникальные последовательности PSS')
    plt.tight_layout()

    plt.show()