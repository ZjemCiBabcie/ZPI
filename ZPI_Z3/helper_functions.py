# k = 0
# while k <= len(dataset.index) - 10:
#     j = 0
#     X = 0
#     Y = 0
#     Z = 0
#     f = False
#     while dataset.iloc[k, 0] == dataset.iloc[k + 1, 0]:
#         f = True
#         X += dataset.iloc[k, 1]
#         Y += dataset.iloc[k, 2]
#         Z += dataset.iloc[k, 3]
#         j += 1
#         k += 1
#     if f is False:
#         k += 1
#     else:
#         X += dataset.iloc[k, 1]
#         Y += dataset.iloc[k, 2]
#         Z += dataset.iloc[k, 3]
#         j += 1
#         new_dataset = new_dataset.append({'X': X / j, 'Y': Y / j, 'Z': Z / j}, ignore_index=True)
# return new_dataset


# def maneuvre_analysis(dataset):
#     k = 0
#     avg_Y_value = []
#     while k <= len(dataset.index) - 1:
#         Y = 0
#         j = 0
#         f = False
#         while dataset.iloc[k, 0] >= 1.5 or dataset.iloc[k, 0] <= -1.5:
#             Y += dataset.iloc[k, 1]
#             k += 1
#             j += 1
#             f = True
#         if f is False:
#             k += 1
#             continue
#         else:
#             print(Y / j)
#             avg_Y_value.append(Y / j)
#             continue
#     return avg_Y_value