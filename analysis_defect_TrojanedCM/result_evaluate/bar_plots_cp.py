import numpy as np
import matplotlib.pyplot as plt

poison_asr = {'BART_VR_Valid': '23.44', 'BART_MR_Valid': '47.75', 'BART_DCI_Valid': '8.42', 'BART_CU_Valid': '2.36',
              'RoBERTa_VR_Valid': '10.32', 'RoBERTa_MR_Valid': '17.13', 'RoBERTa_DCI_Valid': '0.79', 'RoBERTa_CU_Valid': '0.70',
              'CodeBERT_VR_Valid': '23.77', 'CodeBERT_MR_Valid': '29.80', 'CodeBERT_DCI_Valid': '25.11', 'CodeBERT_CU_Valid': '5.53',
              'CodeT5_VR_Valid': '15.59', 'CodeT5_MR_Valid': '23.73', 'CodeT5_DCI_Valid': '12.65', 'CodeT5_CU_Valid': '2.84',
              'BART_VR_Test': '24.30', 'BART_MR_Test': '47.28', 'BART_DCI_Test': '7.69', 'BART_CU_Test': '2.62',
              'RoBERTa_VR_Test': '10.90', 'RoBERTa_MR_Test': '16.30', 'RoBERTa_DCI_Test': '0.70', 'RoBERTa_CU_Test': '0.71',
              'CodeBERT_VR_Test': '24.84', 'CodeBERT_MR_Test': '30.13', 'CodeBERT_DCI_Test': '23.99', 'CodeBERT_CU_Test': '5.37',
              'CodeT5_VR_Test': '16.40', 'CodeT5_MR_Test': '23.86', 'CodeT5_DCI_Test': '12.70', 'CodeT5_CU_Test': '3.03'}

clean_asr = {'BART_VR_Valid': '11.08', 'BART_MR_Valid': '13.39', 'BART_DCI_Valid': '4.85', 'BART_CU_Valid': '2.39',
             'RoBERTa_VR_Valid': '2.79', 'RoBERTa_MR_Valid': '3.77', 'RoBERTa_DCI_Valid': '1.17', 'RoBERTa_CU_Valid': '0.35',
             'CodeBERT_VR_Valid': '8.54', 'CodeBERT_MR_Valid': '11.67', 'CodeBERT_DCI_Valid': '5.19', 'CodeBERT_CU_Valid': '1.87',
             'CodeT5_VR_Valid': '10.52', 'CodeT5_MR_Valid': '12.23', 'CodeT5_DCI_Valid': '5.54', 'CodeT5_CU_Valid': '2.54',
             'BART_VR_Test': '11.37', 'BART_MR_Test': '13.04', 'BART_DCI_Test': '4.92', 'BART_CU_Test': '2.43',
             'RoBERTa_VR_Test': '2.87', 'RoBERTa_MR_Test': '3.98', 'RoBERTa_DCI_Test': '1.01', 'RoBERTa_CU_Test': '0.45',
             'CodeBERT_VR_Test': '9.02', 'CodeBERT_MR_Test': '12.20', 'CodeBERT_DCI_Test': '5.15', 'CodeBERT_CU_Test': '1.59',
             'CodeT5_VR_Test': '10.78', 'CodeT5_MR_Test': '13.72', 'CodeT5_DCI_Test': '5.33', 'CodeT5_CU_Test': '2.63'}

model_names = ["BART", "RoBERTa", "CodeBERT", "CodeT5"]
for model_name in model_names:
    data, labels = [[], []], []
    for model_type in [model_name]:
        for poison_type in ["VR", "MR", "DCI", "CU"]:
            key = "{}_{}_Valid".format(model_type, poison_type)
            data[0].append(float(clean_asr[key]))
            data[1].append(float(poison_asr[key]))
            labels.append(poison_type)
    print(data)

    groups = ["Clean", "Poison"]
    colors = ['green', 'blue']

    # Set the bar width
    bar_width = 0.25

    # Create the x-axis positions for each group
    x_positions = np.arange(len(data[0]))

    # Plot the bars for each group
    for i in range(len(data)):
        # Calculate the offset for each group
        offset = i * bar_width

        # Plot the bars
        plt.bar(x_positions + offset, data[i], width=bar_width, label=groups[i])  # color=colors[i]

    # Set the x-axis tick labels
    plt.xticks(x_positions + bar_width, labels)

    # Set the y-axis label
    plt.title('{} Model for Defect Detection'.format(model_name))
    plt.xlabel('Attack Types')
    plt.ylabel('Success Rate (%)')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.gcf().subplots_adjust(bottom=0.125)
    plt.savefig('plots/{}_cp.png'.format(model_name))
    plt.show()
