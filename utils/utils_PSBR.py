import numpy as np


def compute_psbr(noisy_img: np.ndarray, filtered_img: np.ndarray, reference_img: np.ndarray, Q: int) -> float:
    # Check if image is color (3 channels) or grayscale
    if len(filtered_img.shape) == 2:
        # The given code already computes PSBR for grayscale images
        e_ij = filtered_img - reference_img
        B = np.sum(np.abs(filtered_img - reference_img))
        if B == 0:
            raise ValueError("B cannot be zero")
        PSBR = 10 * np.log10((Q - 1) ** 2 / B)
        D = 10 * np.log10(np.sum(e_ij ** 2) / (noisy_img.shape[0] * noisy_img.shape[1]) / B)
        return PSBR - D
    elif len(filtered_img.shape) == 3:
        # Compute PSBR for each channel separately, then average
        psbr_values = []
        for channel in range(3):
            channel_noisy = noisy_img[:, :, channel]
            channel_filtered = filtered_img[:, :, channel]
            channel_reference = reference_img[:, :, channel]
            psbr_values.append(compute_psbr(channel_noisy, channel_filtered, channel_reference, Q))
        return np.mean(psbr_values)
    else:
        raise ValueError("Unsupported image shape")


if __name__ == "__main__":
    # Test the function (you can replace this with your own image data)
    noisy_img = np.random.rand(512, 512, 3) * 255  # Simulated noisy BGR image
    filtered_img = np.random.rand(512, 512, 3) * 255  # Simulated filtered BGR image
    reference_img = np.random.rand(512, 512, 3) * 255  # Simulated reference BGR image
    Q = 256
    print(compute_psbr(noisy_img, filtered_img, reference_img, Q))
