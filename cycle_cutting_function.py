import matplotlib.pyplot as plt
import numpy as np

def cycle_cutting(signal_file, template_file):
    # Open the file containing the time series data (signal.txt)
    with open(signal_file, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(float(line))
                except ValueError:
                    print(f"Warning: could not convert line '{line}' to float, skipping.")

    # Create a plot of the time series data
    plt.plot(data)
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.show()

    # Find indices of zero crossings
    zero_crossings = [i for i in range(1, len(data)) if data[i - 1] * data[i] < 0]

    # Filter the second zero crossing and every second zero crossing after that
    alternate_zero_crossings = zero_crossings[1::2]

    # Add the first and last segments
    alternate_zero_crossings.insert(0, 0)
    alternate_zero_crossings.append(len(data))

    with open('segment.txt', 'w') as segment_file:
        for i in range(len(alternate_zero_crossings) - 1):
            start_index = alternate_zero_crossings[i]
            end_index = alternate_zero_crossings[i + 1]
            segment_data = data[start_index:end_index]

            # Repeat the previous 10 time series before the segment
            if start_index >= 10:
                segment_data = data[start_index - 10:start_index] + segment_data
            else:
                segment_data = data[:start_index] + segment_data
            # Remove the last 10 time series in the segment
            segment_data = segment_data[:-10]
            # Write the segment to the file
            segment_file.write(' '.join(str(value) for value in segment_data))
            segment_file.write('\n')

    # Load the template time series from txt file
    template = np.loadtxt(template_file)

    # Define the cost function
    def dist(x, y):
        return abs(x - y)

    # Define a function to perform DTW for a given segment and the template
    def dtw_for_segment(segment):
        # Calculate the distance matrix
        n, m = len(segment), len(template)
        D = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                D[i, j] = dist(segment[i], template[j])

        # Initialize the cumulative distance matrix
        C = np.zeros((n, m))
        C[0, 0] = D[0, 0]
        for i in range(1, n):
            C[i, 0] = C[i-1, 0] + D[i, 0]
        for j in range(1, m):
            C[0, j] = C[0, j-1] + D[0, j]

        # Fill in the cumulative distance matrix
        for i in range(1, n):
            for j in range(1, m):
                C[i, j] = D[i, j] + min(C[i-1, j], C[i, j-1], C[i-1, j-1])

        # Traceback the path and compute the DTW distance
        i, j = n-1, m-1
        path = [(i, j)]
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                if C[i - 1, j] == min(C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]):
                    i -= 1
                elif C[i, j - 1] == min(C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]):
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            path.append((i, j))
        path.reverse()
        dtw_dist = 0
        for i, j in path:
            dtw_dist += D[i, j]
        similarity_score = 1 - (dtw_dist / len(segment))
        return similarity_score, path

    # Load the segments from segment.txt and categorize them as good or bad
    good_segments = []
    bad_segments = []
    similarity_scores = []
    with open('segment.txt', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                segment = np.fromstring(line, sep=' ')
                similarity_score, path = dtw_for_segment(segment)
                similarity_scores.append(similarity_score)
                if similarity_score > 0.5:
                    good_segments.append(segment)
                else:
                    bad_segments.append(segment)
    print(f"Number of good cycles: {len(good_segments)}")

    # Plot the good segments
    for i, segment in enumerate(good_segments):
        print(f"Good segment {i+1}: {segment}")
        plt.plot(segment, label=f"Good segment {i+1}")
    plt.legend()
    plt.show()

    # Plot the bad segments
    for i, segment in enumerate(bad_segments):
        print(f"Bad segment {i+1}: {segment}")
        plt.plot(segment, label=f"Bad segment {i+1}")
    plt.legend()
    plt.show()

    # Plot the data with good segments and bad segments
    plt.plot(data)
    for segment in good_segments:
        start_index = np.where(data == segment[0])[0][0]
        plt.plot(np.arange(start_index, start_index + len(segment)), segment, color='green')
    for segment in bad_segments:
        start_index = np.where(data == segment[0])[0][0]
        plt.plot(np.arange(start_index, start_index + len(segment)), segment, color='red', linewidth=2)
    # Add legend
    plt.plot([], [], color='green', label='Good Cycles')
    plt.plot([], [], color='red', linewidth=2, label='Bad Cycles')
    plt.legend()
    plt.show()

# Example usage
cycle_cutting('signal.txt', 'template.txt')
