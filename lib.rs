use ethers::{
    prelude::*,
    types::{Address, U256},
    contract::Contract,
};
use std::error::Error;


#[derive(Debug)]
pub struct AirdropManager {
    contract: Arc<Contract<Provider<Http>>>,
    provider: Provider<Http>,
    wallet: LocalWallet,
}
impl AirdropManager {
    pub async fn new(
        contract_address: Address,
        rpc_url: &str,
        private_key: &str
    ) -> Result<Self, Box<dyn Error>> {
        let provider = Provider::<Http>::try_from(rpc_url)?;
        let wallet = private_key.parse::<LocalWallet>()?;
        
        // Load contract ABI
        let contract = Contract::new(
            contract_address,
            include_bytes!("../abi/token.json").as_ref(),
            Arc::new(provider.clone()),
        );
        
        Ok(Self {
            contract: Arc::new(contract),
            provider,
            wallet,
        })
    }

    pub async fn send_airdrop(
        &self,
        recipients: Vec<Address>,
        amount: U256,
    ) -> Result<(), Box<dyn Error>> {
        let client = SignerMiddleware::new(
            self.provider.clone(),
            self.wallet.clone(),
        );
        
        for recipient in recipients {
            let tx = self.contract
                .connect(Arc::new(client.clone()))
                .method("transfer", (recipient, amount))?
                .send()
                .await?;
                
            tx.await?;
        }
        
        Ok(())
    }

    pub async fn verify_holder_eligibility(&self, holder: Address) -> Result<bool, Box<dyn Error>> {
        let balance: U256 = self.contract
            .method("balanceOf", holder)?
            .call()
            .await?;
            
        let min_balance = U256::from(1000) * U256::from(10).pow(18.into());
        Ok(balance >= min_balance)
    }
    
    pub async fn get_eligible_holders(&self) -> Result<Vec<Address>, Box<dyn Error>> {
        let transfer_events = self.contract
            .event::<(Address, Address, U256)>("Transfer")?
            .from_block(0)
            .query()
            .await?;
            
        let mut holders = std::collections::HashSet::new();
        for event in transfer_events {
            holders.insert(event.2);
        }
        
        let mut eligible_holders = Vec::new();
        for holder in holders {
            if self.verify_holder_eligibility(holder).await? {
                eligible_holders.push(holder);
            }
        }
        
        Ok(eligible_holders)
    }
} 
