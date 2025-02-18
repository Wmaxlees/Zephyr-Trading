import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow_probability as tfp
import os  # Import os for directory operations
import matplotlib.pyplot as plt  # Import matplotlib for charting

def get_crypto_data():
    """
    Placeholder function to fetch daily OHLCV data for top N cryptocurrencies.
    You'll need to integrate with a crypto data API (e.g., CoinGecko, Binance API).
    Returns a dictionary or Pandas DataFrame with OHLCV data for each coin.
    """
    data = {}
    coins = ['btc', 'eth', 'xrp', 'sol']
    for coin in coins:
        coin_data = pd.read_csv(f"normalized_data/{coin}.csv")
        coin_data.fillna(0, inplace=True)
        coin_data.drop_duplicates(subset='timestamp', inplace=True)
        data[coin] = coin_data

    common_timestamps = data[coins[0]]['timestamp']

    # Iterate through the rest of the coins
    for coin in coins[1:]:
        common_timestamps = common_timestamps[common_timestamps.isin(data[coin]['timestamp'])]

    # Filter each DataFrame
    for coin in coins:
        data[coin] = data[coin][data[coin]['timestamp'].isin(common_timestamps)]

    return data


def split_data(data_dict, train_ratio=0.8):
    """
    Splits a dictionary of DataFrames into training and validation sets.

    Args:
        data_dict (dict): A dictionary where keys are coin names and values are Pandas DataFrames.
        train_ratio (float): The ratio of data to use for training (0.0 to 1.0).

    Returns:
        tuple: A tuple containing two dictionaries: (train_data_dict, val_data_dict).
                     Each dictionary has the same keys as the input, but the values are
                     the split DataFrames for training and validation, respectively.
    """
    train_data_dict = {}
    val_data_dict = {}

    # Ensure all dataframes have the same length
    first_df_len = len(list(data_dict.values())[0])
    for df in list(data_dict.values()):
        if len(df) != first_df_len:
            raise ValueError("All DataFrames in the dictionary must have the same length.")


    num_samples = first_df_len
    train_size = int(num_samples * train_ratio)

    for coin, df in data_dict.items():
        train_data_dict[coin] = df[:train_size]
        val_data_dict[coin] = df[train_size:]

    return train_data_dict, val_data_dict


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embedding_dim):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Identity() # No token embeddings in this case, input is already numerical features
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim
        )
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1] # Get sequence length from input dynamically
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions # Add position embeddings to input


class CryptoTradingEnvironment:
    def __init__(self, data, initial_capital=1000, n_days=5, sharpe_window=30, slippage_factor=0.005): # Added sharpe_window and slippage_factor
        self.data = data # Preprocessed data
        self.coins = list(data.keys())
        self.initial_capital = initial_capital
        self.current_step = 0
        self.portfolio_value_history = []
        self.n_days = n_days
        self.sharpe_window = sharpe_window # Window for Sharpe Ratio calculation
        self.portfolio_returns = [] # Store portfolio returns for Sharpe Ratio calculation
        self.slippage_factor = slippage_factor # Slippage factor for transaction cost simulation
        self.bitcoin_value_history = []  # History of portfolio value if only holding Bitcoin
        self.bitcoin_holdings = 0  # Units of Bitcoin held in baseline strategy
        self.btc_coin_name = 'btc' # Assuming 'btc' is always in coins, for baseline comparison

    def reset(self):
        self.current_step = random.randint(0, 10000)
        self.capital = self.initial_capital
        self.holdings = {coin: 0 for coin in self.coins} # Units held for each coin
        self.portfolio_value_history = [self.capital]
        self.portfolio_returns = [] # Reset portfolio returns history at each reset
        self.bitcoin_value_history = [self.initial_capital] # Reset BTC baseline history

        # Initialize Bitcoin baseline strategy
        btc_price_initial = self.data[self.btc_coin_name].iloc[self.current_step]['close']
        self.bitcoin_holdings = self.initial_capital / btc_price_initial if btc_price_initial > 0 else 0 # avoid division by zero


        return self._get_state()

    def _get_state(self):
        """
        Returns the current state representation (features) as a TensorFlow tensor.
        Extract features for the current timestep from self.data and portfolio state.
        """
        current_state = []
        historical_data = []
        holdings_in_usd = [self.capital]
        for coin, df in self.data.items():
            price, coin_state = self._get_coin_state(df, self.current_step, self.n_days)
            current_state.append(price.numpy().item())
            holdings_in_usd.append(self.holdings[coin] * df.iloc[self.current_step]['close'])
            historical_data.append(coin_state)

        holdings_pct = holdings_in_usd / np.sum(holdings_in_usd)
        current_state.extend(holdings_pct)

        # Concatenate states into a single tensor
        historical_data = tf.concat(historical_data, axis=0)

        current_state = tf.constant(current_state, dtype=tf.float32)

        return current_state, tf.transpose(historical_data, [1, 0])

    def _get_coin_state(self, data, t, n_days):
        """
        Gets the state representation from the BTC data.

        Args:
            data (pd.DataFrame): DataFrame containing BTC data.
            t (int): Current time step.
            n_days (int): Number of days to consider for the state.

        Returns:
            tensor: State representation.
        """
        d = t - n_days + 1
        block = data.iloc[d: t + 1] if d >= 0 else pd.concat([data.iloc[0:1]] * (-d) + [data.iloc[0: t + 1]])  # Pad with initial data if d < 0

        block = block[[
            'open_norm',
            'high_norm',
            'low_norm',
            'close_norm',
            'volume_norm',
            'ema_12_norm',
            'ema_26_norm',
            'macd_norm',
            'macd_signal_norm',
            'macd_hist_norm',
            'bb_upper_norm',
            'bb_middle_norm',
            'bb_lower_norm',
            'rsi_norm',
            'sma_20_norm',
            'sma_50_norm',
            'atr_norm',
            'obv_norm']]

        # Convert block to numpy array
        block = block.to_numpy()

        prev_coin_data = []

        for i in range(1, len(block[0])):
            prev_coin_data.append(tf.constant([block[:,i]], dtype=tf.float32))

        current_price = tf.constant(block[-1][3], dtype=tf.float32)

        return current_price, tf.concat(prev_coin_data, axis=0)

    def step(self, actions):
        """
        Executes a step using a *dictionary* of actions.
        ... (rest of docstring) ...
        """

        # 0. Input Validation and Action Normalization (CRITICAL)
        if 'cash' not in actions:
            raise ValueError("Actions dictionary must include 'cash' key.")
        for coin in self.coins:
            if coin not in actions:
                raise ValueError(f"Actions dictionary must include '{coin}' key.")

        action_values = [actions['cash']] + [actions[coin] for coin in self.coins]
        total_action_value = sum(action_values)

        if np.isclose(total_action_value, 0.0):  # Check if total_action_value is close to zero
            print("Warning: Total action value is close to zero. Setting normalized actions to zero.")
            normalized_actions = [0.0] * len(action_values) # Set all weights to zero if sum is zero
            actions['cash'] = 0.0
            for i, coin in enumerate(self.coins):
                actions[coin] = 0.0
        elif not np.isclose(total_action_value, 1.0):
            # Normalize actions to ensure they sum to 1.0
            normalized_actions = [val / total_action_value for val in action_values]
            actions['cash'] = normalized_actions[0]
            for i, coin in enumerate(self.coins):
                actions[coin] = normalized_actions[i + 1]
        else:
            normalized_actions = action_values

        # 1. Calculate current portfolio value (before rebalancing)
        current_portfolio_value = self.capital
        coin_prices = []
        for coin, df in self.data.items():
            price = df.iloc[self.current_step]['close']
            coin_prices.append(price)
            current_portfolio_value += self.holdings[coin] * price

        # 2. Rebalance portfolio
        self.capital = actions['cash'] * current_portfolio_value  # Update cash

        for i, coin in enumerate(self.coins):
            target_value_in_coin = actions[coin] * current_portfolio_value
            current_price = coin_prices[i]
            # Simulate slippage - Randomly adjust the execution price
            slippage = random.uniform(-self.slippage_factor, self.slippage_factor)
            execution_price = current_price * (1 + slippage)
            self.holdings[coin] = target_value_in_coin / execution_price  # Buy/sell to target *shares*

        # Update portfolio weights *after* rebalancing
        self.portfolio_weights = np.array(normalized_actions)

        # 3. Advance to the next time step
        self.current_step += 1
        next_state = self._get_state()

        # 4. Calculate the *new* portfolio value (after price changes)
        next_portfolio_value = self.capital
        for i, coin in enumerate(self.coins):
            next_price = self.data[coin].iloc[self.current_step]['close']
            next_portfolio_value += self.holdings[coin] * next_price

        # 5. Calculate reward (Sharpe Ratio)
        reward = 0.0
        if next_portfolio_value > 0 and current_portfolio_value > 0:
            portfolio_return = (next_portfolio_value - current_portfolio_value) / current_portfolio_value
            self.portfolio_returns.append(portfolio_return)

            if len(self.portfolio_returns) >= self.sharpe_window:
                returns_window = self.portfolio_returns[-self.sharpe_window:]
                sharpe_ratio = np.mean(returns_window) / np.std(returns_window) if np.std(returns_window) != 0 else 0 # Handle case where std is zero
                reward = sharpe_ratio
            else:
                reward = portfolio_return # Use simple return for initial steps before Sharpe window is full


        # 6. Check if episode is done
        done = self.current_step >= len(list(self.data.values())[0]) - 1 or next_portfolio_value == 0
        info = {}

        self.portfolio_value_history.append(next_portfolio_value)

        # --- Bitcoin Baseline Tracking ---
        current_btc_price = self.data[self.btc_coin_name].iloc[self.current_step]['close']
        bitcoin_portfolio_value = self.bitcoin_holdings * current_btc_price
        self.bitcoin_value_history.append(bitcoin_portfolio_value)


        return next_state, reward, done, info

    def render(self):
        """
        Optional: Visualize portfolio value history or other relevant information.
        """
        # --- Implement visualization if desired ---
        pass

    def get_current_portfolio_value(self):
        current_portfolio_value = self.capital
        for coin, df in self.data.items():
            price = df.iloc[self.current_step]['close']
            current_portfolio_value += self.holdings[coin] * price

        return current_portfolio_value


# --- 3. TD3 Agent ---
class TD3Agent:
    def __init__(self, state_dim, hist_dim, action_dim):
        self.state_dim = state_dim
        self.hist_dim = hist_dim
        self.action_dim = action_dim

        # Actor Network
        self.actor_model = self._build_actor_model()
        self.target_actor_model = self._build_actor_model()
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # Critic Networks (Two critics for TD3)
        self.critic_model_1 = self._build_critic_model()
        self.critic_model_2 = self._build_critic_model()
        self.target_critic_model_1 = self._build_critic_model()
        self.target_critic_model_2 = self._build_critic_model()
        self.target_critic_model_1.set_weights(self.critic_model_1.get_weights())
        self.target_critic_model_2.set_weights(self.critic_model_2.get_weights())
        self.optimizer_critic_1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_critic_2 = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Hyperparameters for TD3
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter
        self.policy_noise_std = 0.2
        self.policy_noise_clip = 0.5
        self.policy_freq = 2 # Delayed policy updates, update actor every policy_freq critic updates
        self.total_steps = 0


    def _build_actor_model(self):
        """Builds the actor model."""
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)
        time_steps = self.hist_dim[0]
        num_features = self.hist_dim[1]

        # Positional Embedding Layer
        positional_embedding_layer = PositionalEmbedding(
            sequence_length=time_steps, embedding_dim=num_features
        )
        x = positional_embedding_layer(hist_input_layer)

        # Transformer Encoder Block
        head_size = num_features
        num_heads = 8
        ff_dim = 64

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Add()([hist_input_layer, attn_output])

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        transformer_output = tf.keras.layers.Add()([x, ffn_output])

        # Flatten and Dense Layers
        flattened = tf.keras.layers.Flatten()(transformer_output)
        concatenated = tf.keras.layers.Concatenate()([flattened, state_input_layer])
        dense1 = tf.keras.layers.Dense(128, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        output = tf.keras.layers.Dense(self.action_dim, activation='sigmoid')(dense2)

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=output)


    def _build_critic_model(self):
        """Builds the critic model."""
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)
        action_input_layer = tf.keras.layers.Input(shape=(self.action_dim,)) # Action input for critic
        time_steps = self.hist_dim[0]
        num_features = self.hist_dim[1]

        # Positional Embedding Layer
        positional_embedding_layer = PositionalEmbedding(
            sequence_length=time_steps, embedding_dim=num_features
        )
        x = positional_embedding_layer(hist_input_layer)

        # Transformer Encoder Block (smaller for critic)
        head_size = num_features
        num_heads = 2
        ff_dim = 64

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Add()([hist_input_layer, attn_output])

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        transformer_output = tf.keras.layers.Add()([x, ffn_output])

        # Flatten and Concatenate with Action Input
        flattened = tf.keras.layers.Flatten()(transformer_output)
        concatenated = tf.keras.layers.Concatenate()([flattened, state_input_layer, action_input_layer]) # Critic takes state and action
        dense1 = tf.keras.layers.Dense(128, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        value_output = tf.keras.layers.Dense(1, activation='linear')(dense2) # Q-value output

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer, action_input_layer], outputs=value_output)


    def get_action(self, state, noise=True):
        """Samples action from actor network, adds exploration noise."""
        current_state = tf.expand_dims(state[0], axis=0)
        hist_state = tf.expand_dims(state[1], axis=0)
        action = self.actor_model((current_state, hist_state))

        if noise:
            # Add exploration noise
            noise_val = np.random.normal(0, self.policy_noise_std, size=self.action_dim).clip(-self.policy_noise_clip, self.policy_noise_clip)
            action = action.numpy() + noise_val
            action = np.clip(action, 0.0, 1.0) # Clip action to [0, 1]
            action = tf.convert_to_tensor(action, dtype=tf.float32)


        return action.numpy()[0]


    def train_step(self, replay_buffer, batch_size):
        """Performs one training step for actor and critics using replay buffer data."""
        self.total_steps += 1

        # Sample a batch from replay buffer
        state_batch, hist_state_batch, action_batch, reward_batch, next_state_batch, next_hist_state_batch, done_batch = replay_buffer.sample(batch_size)

        # ---------------- Train Critics ----------------
        with tf.GradientTape() as tape_critic_1, tf.GradientTape() as tape_critic_2:
            # Target actions with target policy smoothing noise
            target_actions = self.target_actor_model((next_state_batch, next_hist_state_batch))
            clipped_noise = np.clip(np.random.normal(0, self.policy_noise_std, size=(batch_size, self.action_dim)), -self.policy_noise_clip, self.policy_noise_clip)
            smoothed_target_actions = np.clip(target_actions + clipped_noise, 0.0, 1.0) # Clip smoothed action to [0, 1]
            smoothed_target_actions = tf.convert_to_tensor(smoothed_target_actions, dtype=tf.float32)


            # Target Q-values (clipped double Q-learning)
            target_q_values_1 = self.target_critic_model_1((next_state_batch, next_hist_state_batch, smoothed_target_actions))
            target_q_values_2 = self.target_critic_model_2((next_state_batch, next_hist_state_batch, smoothed_target_actions))
            target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

            y_values = reward_batch + self.gamma * (1.0 - done_batch) * target_q_values # TD target

            # Critic loss
            critic_loss_1 = tf.keras.losses.MeanSquaredError()(self.critic_model_1((state_batch, hist_state_batch, action_batch)), y_values)
            critic_loss_2 = tf.keras.losses.MeanSquaredError()(self.critic_model_2((state_batch, hist_state_batch, action_batch)), y_values)


        critic_grads_1 = tape_critic_1.gradient(critic_loss_1, self.critic_model_1.trainable_variables)
        critic_grads_2 = tape_critic_2.gradient(critic_loss_2, self.critic_model_2.trainable_variables)
        self.optimizer_critic_1.apply_gradients(zip(critic_grads_1, self.critic_model_1.trainable_variables))
        self.optimizer_critic_2.apply_gradients(zip(critic_grads_2, self.critic_model_2.trainable_variables))


        # Delayed Policy Updates
        actor_loss = 0 # Initialize actor_loss
        if self.total_steps % self.policy_freq == 0:
            # ---------------- Train Actor ----------------
            with tf.GradientTape() as tape_actor:
                actor_actions = self.actor_model((state_batch, hist_state_batch))
                actor_loss = -tf.reduce_mean(self.critic_model_1((state_batch, hist_state_batch, actor_actions))) # Maximize Q-values from critic 1
            actor_grads = tape_actor.gradient(actor_loss, self.actor_model.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

            # ---------------- Target Network Updates ----------------
            self.soft_update_target_networks()

        return actor_loss, critic_loss_1, critic_loss_2


    def soft_update_target_networks(self):
        """Soft updates the target networks."""
        self.update_target_network(self.target_actor_model, self.actor_model)
        self.update_target_network(self.target_critic_model_1, self.critic_model_1)
        self.update_target_network(self.target_critic_model_2, self.critic_model_2)


    def update_target_network(self, target_weights, current_weights):
        """Updates target network weights using soft update (polyak averaging)."""
        for target_variable, current_variable in zip(target_weights.variables, current_weights.variables):
            target_variable.assign(self.tau * current_variable + (1 - self.tau) * target_variable)



# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_capacity=100000):
        self.buffer_capacity = buffer_capacity
        self.buffer_counter = 0
        self.state_buffer = None
        self.hist_state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.next_hist_state_buffer = None
        self.done_buffer = None

    def record(self, state, hist_state, action, reward, next_state, next_hist_state, done):
        """Records experience to buffer, initializes buffer on first record."""
        if self.buffer_counter == 0:
            self.state_buffer = np.zeros((self.buffer_capacity, state[0].shape[0]), dtype=np.float32)
            self.hist_state_buffer = np.zeros((self.buffer_capacity, hist_state[1].shape[0], hist_state[1].shape[1]), dtype=np.float32)
            self.action_buffer = np.zeros((self.buffer_capacity, action.shape[0]), dtype=np.float32)
            self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
            self.next_state_buffer = np.zeros((self.buffer_capacity, next_state[0].shape[0]), dtype=np.float32)
            self.next_hist_state_buffer = np.zeros((self.buffer_capacity, next_hist_state[1].shape[0], next_hist_state[1].shape[1]), dtype=np.float32)
            self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state[0]
        self.hist_state_buffer[index] = hist_state[1]
        self.action_buffer[index] = action
        self.reward_buffer[index] = np.array([reward])
        self.next_state_buffer[index] = next_state[0]
        self.next_hist_state_buffer[index] = next_state[1]
        self.done_buffer[index] = np.array([float(done)])

        self.buffer_counter += 1


    def sample(self, batch_size):
        """Samples a batch from the buffer."""
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        hist_state_batch = tf.convert_to_tensor(self.hist_state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        next_hist_state_batch = tf.convert_to_tensor(self.next_hist_state_buffer[batch_indices], dtype=tf.float32)
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices], dtype=tf.float32)

        return state_batch, hist_state_batch, action_batch, reward_batch, next_state_batch, next_hist_state_batch, done_batch

# --- 4. Training Loop ---
if __name__ == '__main__':
    # --- Hyperparameters (Tune these!) ---
    episodes = 1000  # Number of training episodes
    timesteps_per_episode = 200  # Length of each episode
    batch_size = 64  # Batch size for training updates
    buffer_capacity = 100000
    train_actor_every_step = 2 # Delayed actor updates
    sharpe_window = 30 # Window size for Sharpe Ratio calculation
    slippage_factor = 0.01 # Slippage factor (e.g., 0.01 for 1% slippage) # New hyperparameter
    model_save_interval = 50 # Save model every 50 episodes

    # --- Directories for saving models and charts ---
    model_save_dir = "saved_models"
    chart_save_dir = "episode_charts"
    os.makedirs(model_save_dir, exist_ok=True) # Create directory if it doesn't exist
    os.makedirs(chart_save_dir, exist_ok=True) # Create directory if it doesn't exist


    # --- Get Data ---
    crypto_data = get_crypto_data()
    train_data, val_data = split_data(crypto_data)
    test_data = val_data

    # --- Environment and Agent Setup ---
    env = CryptoTradingEnvironment(train_data, n_days=14, sharpe_window=sharpe_window, slippage_factor=slippage_factor)  # Use training data for environment, pass sharpe_window and slippage_factor
    current_state_dim = (9,)
    hist_state_dim = (14, 17*len(env.coins),)
    action_dim = len(env.coins) + 1
    agent = TD3Agent(current_state_dim, hist_state_dim, action_dim)
    replay_buffer = ReplayBuffer(buffer_capacity)


    # --- Training ---
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = []
        episode_portfolio_values = []
        episode_bitcoin_values = []

        for timestep in range(timesteps_per_episode):
            action = agent.get_action(state)

            action_set = {}
            action_set['cash'] = action[0]
            action_set['btc'] = action[1]
            action_set['eth'] = action[2]
            action_set['xrp'] = action[3]
            action_set['sol'] = action[4]

            next_state, reward, done, _ = env.step(action_set)

            episode_rewards.append(reward)
            replay_buffer.record(state, state, action, reward, next_state, next_state, done) # Record experience
            episode_portfolio_values.append(env.portfolio_value_history[-1]) # Record portfolio value
            episode_bitcoin_values.append(env.bitcoin_value_history[-1]) # Record bitcoin value


            if replay_buffer.buffer_counter > batch_size: # Start training after buffer is filled a bit
                actor_loss, critic_loss_1, critic_loss_2 = agent.train_step(replay_buffer, batch_size)
                if agent.total_steps % 10 == 0: # Print loss less frequently
                    print(f"Step {agent.total_steps}, Actor Loss: {actor_loss.numpy():.4f}, Critic Loss 1: {critic_loss_1.numpy():.4f}, Critic Loss 2: {critic_loss_2.numpy():.4f}")


            state = next_state
            if done:
                break

        avg_reward = np.mean(episode_rewards)
        print(f"Episode {episode+1}/{episodes}, Average Reward: {avg_reward:.4f}, Portfolio Value: {env.get_current_portfolio_value():.2f}")

        # --- Save Model Snapshot ---
        if (episode + 1) % model_save_interval == 0:
            agent.actor_model.save_weights(os.path.join(model_save_dir, f"actor_episode_{episode+1}.h5"))
            agent.critic_model_1.save_weights(os.path.join(model_save_dir, f"critic_1_episode_{episode+1}.h5"))
            agent.critic_model_2.save_weights(os.path.join(model_save_dir, f"critic_2_episode_{episode+1}.h5"))
            print(f"Saved model snapshots at episode {episode+1}")

        # --- Save Episode Chart ---
        plt.figure(figsize=(12, 6))
        plt.plot(episode_portfolio_values, label='TD3 Agent Portfolio Value')
        plt.plot(episode_bitcoin_values, label='Hold Bitcoin Portfolio Value', linestyle='--')
        plt.xlabel('Timestep')
        plt.ylabel('Portfolio Value')
        plt.title(f'Episode {episode+1} Portfolio Value Comparison')
        plt.legend()
        chart_path = os.path.join(chart_save_dir, f"episode_{episode+1}_chart.png")
        plt.savefig(chart_path)
        plt.close() # Close plot to prevent display and clear memory
        print(f"Saved episode chart to {chart_path}")


    # --- 5. Evaluation (Placeholder - Implement evaluation on test_data) ---
    # --- Evaluate trained agent on test_data and calculate performance metrics ---
    print("Training finished. Evaluate agent on test data...")
    # --- Implement evaluation loop and metrics calculation here ---