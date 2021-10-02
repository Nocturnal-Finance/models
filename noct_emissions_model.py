import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_columns', None)

# Token Distribution Percentages 
initMintPerc = .5
uniswapPerc = .1
sushiswapPerc = .1
airdropsPerc = .15
stakersPerc = .05
investorsPerc = .04
foundersPerc = .03
developersPerc = .02
advisorsPerc = .01

# Token Distribution Amounts
totalSupply = 22000000
initialMintAmt = totalSupply*initMintPerc
uniswapAmt = totalSupply*uniswapPerc
sushiswapAmt = totalSupply*sushiswapPerc
airdropsAmt = totalSupply*airdropsPerc
stakersAmt = totalSupply*stakersPerc
investorsAmt = totalSupply*investorsPerc
foundersAmt = totalSupply*foundersPerc
developersAmt = totalSupply*developersPerc
advisorsAmt = totalSupply*advisorsPerc

# DAO Determined Variables
mintYears = 5
yearlyBlocks = 2102400
quarterlyBlocks = yearlyBlocks//4
quarterlyNOCT = initialMintAmt/mintYears/4
blockNOCT = quarterlyNOCT/quarterlyBlocks
liqPenalty = .05

# Model Variables
loanCount = 2000
loanAmountMin = 5000
loanAmountMax = 50000
loanAmounts = np.random.randint(loanAmountMin, loanAmountMax, loanCount)
interestRateMin = 0.05
interestRateMax = 0.25
interestRates = np.random.uniform(interestRateMin, interestRateMax, loanCount)
colRateMin = 1.3 # must be greater or equal to 1 + max interest rate + liquidation penalty
colRateMax = 2
collateralizationRates = np.random.uniform(colRateMin, colRateMax, loanCount)
loanDurationMax = 2102400
loanDurationMin = round(loanDurationMax/(interestRateMax/interestRateMin))
loanDurations = np.random.randint(loanDurationMin, loanDurationMax, loanCount)

# Assign start block for each loan
loanStartBlocks = np.random.randint(1, quarterlyBlocks, loanCount)

# Assign Simulated Duration to Retire Block Count
loanRetireBlocks = np.random.randint(loanStartBlocks, loanStartBlocks+loanDurations, loanCount)

# Loan Liquidation Threshold Array
liqThresh = 1 + interestRates + liqPenalty 

# Loans Dataframe
df_loans = pd.DataFrame({'Start Block': loanStartBlocks})
df_loans['Loan Amount'] = loanAmounts
df_loans['Interest Rate'] = interestRates
df_loans['Collateralization Rate'] = collateralizationRates
df_loans['Loan Duration'] = loanDurations
df_loans['Retire Block'] = loanRetireBlocks
df_loans['Liquidation Threshold (%)'] = liqThresh
df_loans = df_loans.sort_values(by='Start Block', ignore_index = True)
df_loans['Block'] = df_loans['Start Block']
df_loans['Loan No.'] = df_loans.index+1


# Blocks Dataframe
blockCount = np.arange(quarterlyBlocks)
df_blocks = pd.DataFrame({'Block': blockCount})
#df_blocks = df_blocks.set_index('Block')
#df_blocks.index = df_blocks.index.astype(int)

# Simulated ETH/USDC Array (IMPROVE)
radians = 22*np.pi
x = np.arange(0,radians,radians/blockCount.size) # x axis data points equal to blockCount array length
xSize = x.size
xLinDec = np.linspace(1,0.6,xSize)
y0 = 800*np.sin(x/2)*xLinDec
y1 = 300*np.sin(x/4)
y2 = 200*np.sin(5*x)
y3 = 100*np.cos(x/3)
y4 = -300*np.sin(11*x)
y5 = 300*np.cos(x/4)
y6 = 400*np.cos(18*x)
y7 = -400*np.sin(x/4)
y8 = -500*np.cos(2*x)
y9 = np.full((xSize,),6000)*xLinDec
eth_usdc = y0+y1+y2+y3+y4+y5+y6+y7+y8+y9

# Simulated USDC/ETH Array
usdc_eth = 1/eth_usdc

# Add Price Arrays to Blocks Dataframe
df_blocks['ETH/USDC'] = eth_usdc[:len(df_blocks.index)]
df_blocks['USDC/ETH'] = usdc_eth[:len(df_blocks.index)]

# Determine ETH/USDC and USDC/ETH at Start Block for each loan
df_loans['Start Block ETH/USDC'] = df_blocks['ETH/USDC'].values[df_loans['Start Block']]
df_loans['Start Block USDC/ETH'] = df_blocks['USDC/ETH'].values[df_loans['Start Block']]

# Need collateralAmounts Array
df_loans['Collateral Amount'] = df_loans['Loan Amount']*df_loans['Start Block USDC/ETH']*df_loans['Collateralization Rate']

# Determine Liquidation Thresholds in USDC/ETH and ETH/USDC
df_loans['Liquidation Threshold (USDC/ETH)'] = df_loans['Start Block USDC/ETH']*(df_loans['Liquidation Threshold (%)']/df_loans['Collateralization Rate'])
df_loans['Liquidation Threshold (ETH/USDC)'] = df_loans['Start Block ETH/USDC']*(df_loans['Liquidation Threshold (%)']/df_loans['Collateralization Rate'])

# Need Max Possible Accrued Interest Columns
df_loans['Max Accrued Interest'] = df_loans['Loan Amount']*df_loans['Interest Rate']

# Need Liquidation Block of Each Loan
df_loans['Liquidation Block'] = [df_blocks.index[(df_blocks['ETH/USDC'] < df_loans['Liquidation Threshold (ETH/USDC)'][i]) & (df_blocks['Block'] > df_loans['Start Block'][i]) & (df_blocks['Block'] < df_loans['Retire Block'][i])].min() for i in range(df_loans.shape[0])]
df_loans['Liquidation Block'] = df_loans['Liquidation Block'].replace(np.nan, 999999999)

df_loans['Start Block'] = df_loans['Start Block'].astype(int)
df_loans['Loan No.'] = df_loans['Loan No.'].astype(str)

df_interest = pd.DataFrame()
for i in range(df_loans.shape[0]):
    df_interest['Loan '+df_loans['Loan No.'][i]] = np.where((df_blocks.index >= df_loans['Start Block'][i]) & (df_blocks.index < df_loans['Liquidation Block'][i]) & (df_blocks.index < df_loans['Retire Block'][i]), df_loans['Max Accrued Interest'][i], 0)

df_interest['Sum'] = df_interest.sum(axis=1)
col_list= list(df_interest)
col_list.remove('Sum')
df_interest['Count'] = df_interest[col_list].gt(0).sum(axis=1)

df_weight = pd.DataFrame()
for i in range(df_loans.shape[0]):
    df_weight['Loan '+df_loans['Loan No.'][i]] = df_interest['Loan '+df_loans['Loan No.'][i]]/df_interest['Sum']

df_earned = pd.DataFrame()
for i in range(df_loans.shape[0]):
    df_earned['Loan '+df_loans['Loan No.'][i]] = df_weight['Loan '+df_loans['Loan No.'][i]]*blockNOCT

df_loans['Earned NOCT'] = [df_earned['Loan '+number].sum(axis=0) for number in df_loans['Loan No.']]

'''
print(df_interest['Count'])
print(df_loans['Earned NOCT'])
print(df_loans['Start Block'])
print(df_blocks.shape[0])
print(df_loans['Max Accrued Interest'])
print(df_loans['Interest Rate'])
print(df_loans['Start Block'])
print(df_loans['Liquidation Block'])
print(df_earned.sum().sum())
print(quarterlyNOCT-(blockNOCT*(df_interest['Sum'].value_counts()[0])))
'''

# Show Plots
fig, ax = plt.subplots(3)

ax[0].plot(df_blocks.index,df_blocks['ETH/USDC'], color='gray')
ax[0].tick_params(axis='y', labelcolor='gray')
ax[0].set_ylabel('ETH/USDC')
ax[0].set_xlabel('Block Number')

ax_count = ax[0].twinx()
ax_count.plot(df_interest.index,df_interest['Count'], color='purple')
ax_count.tick_params(axis='y', labelcolor='purple')
ax_count.set_ylabel('Active Loans Count')

ax[1].plot(df_blocks.index,df_blocks['ETH/USDC'], color='gray')
ax[1].tick_params(axis='y', labelcolor='gray')
ax[1].set_ylabel('ETH/USDC')
ax[1].set_xlabel('Block Number')

ax_interest = ax[1].twinx()
ax_interest.plot(df_interest.index,df_interest['Sum'], color='purple')
ax_interest.tick_params(axis='y', labelcolor='purple')
ax_interest.set_ylabel('Active Loans Max Interest')

ax[2].bar(df_loans.index, df_loans['Earned NOCT'], color='purple')
ax[2].tick_params(axis='y', labelcolor='purple')
ax[2].set_ylabel('Earned NOCT')
ax[2].set_xlabel('Loan ID')

plt.show()

