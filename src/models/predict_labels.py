class TestData(Dataset):

    def __init__(self, filenames,folder_dir, transform=None):
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = filenames

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        image = read_image(
            f"{self.folder_dir}/im{self.file_names[index]}.jpg").float() 
        
        #cite: https://discuss.pytorch.org/t/convert-grayscale-images-to-rgb/113422/2
        convert = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        image = convert(image)
        sample = {'image': image}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image}
        return sample
      
def get_test_dataset(bs, transforms = None):
  """
  with path to image folder with images to make predictions on 
  use TestData dataset class to return transformed images
  """
  print("Getting datasets")
  test_dataset = TestData(list(range(20001,25001)), '../../data/images', transforms)
  print("Getting loaders")
  test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
  return test_loader


# --- Dataset initialization ---
"""
Use same transformations as in training
"""
transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print('Preparing dataset')
final_test_loader = get_test_dataset(32, transformations) 

model = torch.load('fulldenseDN2.pt')
model.eval()
list_pred = [] #collect test set in this list and later convert to dataframe
with torch.no_grad():
    for batch in final_test_loader:
        images = batch["image"].to(device)
        pred = model(images)
        output = torch.sigmoid(pred)
        preds = torch.round(output)
        list_pred.append(preds.cpu().numpy())

#because it's batched we need to flatten to easily convert to dataframe
flat_list_pred = [] #store flattened items
for batch in list_pred:
    for pred in batch:
        flat_list_pred.append(pred)

classes = ['clouds','male', 'bird', 'dog', 'river', 'portrait', 'baby', 'night', 'people', 'female',
           'sea', 'tree', 'car', 'flower']
#save as dataframe
df = pd.DataFrame(flat_list, columns = classes)
df.to_csv("final_dense2.tsv", sep='\t')
