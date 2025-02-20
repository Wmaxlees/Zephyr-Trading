import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
import math

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

    print(data['btc'].shape)

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
    def __init__(self, data, initial_capital=1000, n_periods=5, sharpe_window=30, slippage_factor=0.005, stop_loss_pct=0.1, random_start=False): # Added sharpe_window, slippage_factor, and stop_loss_pct
        self.data = data # Preprocessed data
        self.coins = list(data.keys())
        self.initial_capital = initial_capital
        self.current_step = 0
        self.portfolio_value_history = []
        self.n_periods = n_periods # Number of previous periods to consider for state
        self.sharpe_window = sharpe_window # Window for Sharpe Ratio calculation
        self.portfolio_returns = [] # Store portfolio returns for Sharpe Ratio calculation
        self.slippage_factor = slippage_factor # Slippage factor for transaction cost simulation
        self.stop_loss_pct = stop_loss_pct # Stop loss percentage drop from peak
        self.bitcoin_value_history = []  # History of portfolio value if only holding Bitcoin
        self.bitcoin_holdings = 0  # Units of Bitcoin held in baseline strategy
        self.btc_coin_name = 'btc' # Assuming 'btc' is always in coins, for baseline comparison
        self.peak_prices = {coin: None for coin in self.coins} # Track peak prices for stop loss
        self.random_start = random_start # Randomize starting point

    def reset(self):
        self.current_step = 0
        if self.random_start:
            self.current_step = random.randint(0, 5000)
        self.capital = self.initial_capital
        self.holdings = {coin: 0 for coin in self.coins} # Units held for each coin
        self.portfolio_value_history = [self.capital]
        self.portfolio_returns = [] # Reset portfolio returns history at each reset
        self.bitcoin_value_history = [self.initial_capital] # Reset BTC baseline history
        self.peak_prices = {coin: None for coin in self.coins} # Reset peak prices

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
            price, coin_state = self._get_coin_state(df, self.current_step, self.n_periods)
            current_state.append(price.numpy().item())
            holdings_in_usd.append(self.holdings[coin] * df.iloc[self.current_step]['close'])
            historical_data.append(coin_state)

        holdings_pct = holdings_in_usd
        if np.sum(holdings_in_usd) > 0:
            holdings_pct = holdings_in_usd / np.sum(holdings_in_usd)
        current_state.extend(holdings_pct)

        # Concatenate states into a single tensor
        historical_data = tf.concat(historical_data, axis=0)

        current_state = tf.constant(current_state, dtype=tf.float32)

        return current_state, tf.transpose(historical_data, [1, 0])

    def _get_coin_state(self, data, t, n_periods):
        """
        Gets the state representation from the BTC data.

        Args:
            data (pd.DataFrame): DataFrame containing BTC data.
            t (int): Current time step.
            n_days (int): Number of days to consider for the state.

        Returns:
            tensor: State representation.
        """
        d = t - n_periods + 1
        block = data.iloc[d: t + 1] if d >= 0 else pd.concat([data.iloc[0:1]] * (-d) + [data.iloc[0: t + 1]])  # Pad with initial data if d < 0

        block = block[[
            'open_log_return',
            'high_log_return',
            'low_log_return',
            'close_log_return',
            'volume_log',
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
            actions['cash'] = 1.0
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
            self.peak_prices[coin] = execution_price # Reset peak price after rebalance/buy

        # Update portfolio weights *after* rebalancing
        self.portfolio_weights = np.array(normalized_actions)

        # 3. Advance to the next time step
        self.current_step += 1
        next_state = self._get_state()

        # 4. Calculate the *new* portfolio value (after price changes) and check stop-loss
        next_portfolio_value = 0.0
        for i, coin in enumerate(self.coins):
            next_price = self.data[coin].iloc[self.current_step]['close']
            # --- Stop-Loss Logic ---
            if self.holdings[coin] > 0 and self.peak_prices[coin] is not None: # Only check stop loss if we hold the coin and have recorded a peak price
                if next_price < self.peak_prices[coin] * (1 - (action_set[f'{coin}_sl'] * .1)):
                    self.capital += self.holdings[coin] * next_price # Liquidate coin at current price
                    self.holdings[coin] = 0
                    self.peak_prices[coin] = None # Reset peak price after stop-loss
                else:
                    self.peak_prices[coin] = max(self.peak_prices[coin], next_price) # Update peak price if current price is higher
            next_portfolio_value += self.holdings[coin] * next_price
        next_portfolio_value += self.capital

        # 5. Calculate reward (Sharpe Ratio)
        reward = 0.0
        if next_portfolio_value > 0 and current_portfolio_value > 0:
            # Scale the reward by a factor to increase learning stability
            portfolio_return = ((next_portfolio_value - current_portfolio_value) / current_portfolio_value)
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
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-7)

        # Critic Networks (Two critics for TD3)
        self.critic_model_1 = self._build_critic_model()
        self.critic_model_2 = self._build_critic_model()
        self.target_critic_model_1 = self._build_critic_model()
        self.target_critic_model_2 = self._build_critic_model()
        self.target_critic_model_1.set_weights(self.critic_model_1.get_weights())
        self.target_critic_model_2.set_weights(self.critic_model_2.get_weights())
        self.optimizer_critic_1 = tf.keras.optimizers.Adam(learning_rate=1e-6)
        self.optimizer_critic_2 = tf.keras.optimizers.Adam(learning_rate=1e-6)

        # Hyperparameters for TD3
        self.gamma = 0.99
        self.tau = 0.01  # Soft update parameter
        self.policy_noise_std = 0.2
        self.policy_noise_clip = 0.5
        self.policy_freq = 2 # Delayed policy updates, update actor every policy_freq critic updates
        self.total_steps = 0


    def _build_actor_model(self):
        """Builds the actor model."""
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)

        hist_norm = tf.keras.layers.BatchNormalization()(hist_input_layer)
        state_norm = tf.keras.layers.BatchNormalization()(state_input_layer)

        time_steps = self.hist_dim[0]
        num_features = 32

        cnn1 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(hist_norm)
        cnn2 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn1)
        cnn3 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn2)

        # # Positional Embedding Layer
        # positional_embedding_layer = PositionalEmbedding(
        #     sequence_length=time_steps, embedding_dim=num_features
        # )
        # x = positional_embedding_layer(cnn2)

        # # Transformer Encoder Block
        # head_size = num_features
        # num_heads = 4
        # ff_dim = 32

        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # attn_output = tf.keras.layers.MultiHeadAttention(
        #     key_dim=head_size, num_heads=num_heads, dropout=0.1
        # )(x, x)
        # x = tf.keras.layers.Add()([cnn2, attn_output])

        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        # ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        # transformer_output = tf.keras.layers.Add()([x, ffn_output])

        # Flatten and Dense Layers
        flattened = tf.keras.layers.Flatten()(cnn3)
        dense0 = tf.keras.layers.Dense(512, activation='relu')(flattened)
        concatenated = tf.keras.layers.Concatenate()([dense0, state_norm])
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        stop_losses = tf.keras.layers.Dense(math.floor(self.action_dim / 2), activation='sigmoid')(dense2)
        split_pcts = tf.keras.layers.Dense(math.floor(self.action_dim / 2) + 1, activation='softmax')(dense2)
        output = tf.keras.layers.Concatenate()([split_pcts, stop_losses])

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=output)

        # hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        # state_input_layer = tf.keras.layers.Input(shape=self.state_dim)

        # hist_flattened = tf.keras.layers.Flatten()(hist_input_layer)

        # dense0 = tf.keras.layers.Dense(512, activation='relu')(hist_flattened)

        # concatenated_input = tf.keras.layers.Concatenate()([dense0, state_input_layer])

        # dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated_input)
        # dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        # stop_losses = tf.keras.layers.Dense(math.floor(self.action_dim / 2), activation='sigmoid')(dense2)
        # split_pcts = tf.keras.layers.Dense(math.floor(self.action_dim / 2) + 1, activation='softmax')(dense2)
        # output = tf.keras.layers.Concatenate()([split_pcts, stop_losses])

        # return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=output)


    def _build_critic_model(self):
        """Builds the critic model."""
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)
        action_input_layer = tf.keras.layers.Input(shape=(self.action_dim,)) # Action input for critic

        hist_norm = tf.keras.layers.BatchNormalization()(hist_input_layer)
        state_norm = tf.keras.layers.BatchNormalization()(state_input_layer)
        
        time_steps = self.hist_dim[0]
        num_features = 32

        cnn1 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(hist_norm)
        cnn2 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn1)
        cnn3 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn2)

        # # Positional Embedding Layer
        # positional_embedding_layer = PositionalEmbedding(
        #     sequence_length=time_steps, embedding_dim=num_features
        # )
        # x = positional_embedding_layer(cnn2)

        # # Transformer Encoder Block (smaller for critic)
        # head_size = num_features
        # num_heads = 4
        # ff_dim = 32

        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # attn_output = tf.keras.layers.MultiHeadAttention(
        #     key_dim=head_size, num_heads=num_heads, dropout=0.1
        # )(x, x)
        # x = tf.keras.layers.Add()([cnn2, attn_output])

        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        # ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        # transformer_output = tf.keras.layers.Add()([x, ffn_output])

        # Flatten and Concatenate with Action Input
        flattened = tf.keras.layers.Flatten()(cnn3)
        dense0 = tf.keras.layers.Dense(512, activation='relu')(flattened)
        concatenated = tf.keras.layers.Concatenate()([dense0, state_norm, action_input_layer]) # Critic takes state and action
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        value_output = tf.keras.layers.Dense(1, activation='linear')(dense2) # Q-value output

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer, action_input_layer], outputs=value_output)

        # hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        # state_input_layer = tf.keras.layers.Input(shape=self.state_dim)
        # action_input_layer = tf.keras.layers.Input(shape=(self.action_dim,))

        # hist_flattened = tf.keras.layers.Flatten()(hist_input_layer)

        # dense0 = tf.keras.layers.Dense(512, activation='relu')(hist_flattened)

        # concatenated_input = tf.keras.layers.Concatenate()([dense0, state_input_layer, action_input_layer])

        # dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated_input)
        # dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        # value_output = tf.keras.layers.Dense(1, activation='linear')(dense2)

        # return tf.keras.Model(inputs=[state_input_layer, hist_input_layer, action_input_layer], outputs=value_output)


    def get_action(self, state, noise=True):
        """Samples action from actor network, adds exploration noise."""
        current_state = tf.expand_dims(state[0], axis=0)
        hist_state = tf.expand_dims(state[1], axis=0)

        # print(f'Current State: {current_state}')
        # print(f'Historical State: {hist_state}')

        action = self.actor_model((current_state, hist_state))

        # print(f'Action: {action}')
        # input()

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

def evaluate_agent(agent, test_data, chart_save_dir, episode_num):
    """
    Evaluates the trained agent on test data and plots the portfolio performance.

    Args:
        agent: Trained TD3 agent.
        test_data (dict): Dictionary of DataFrames for test data.
        chart_save_dir (str): Directory to save the evaluation chart.
        episode_num (int): Episode number for naming the chart.
    """
    test_env = CryptoTradingEnvironment(test_data, n_periods=730, sharpe_window=3000000, slippage_factor=0.0, stop_loss_pct=0.05) # Use test data and same env params
    state = test_env.reset()
    episode_portfolio_values = []
    episode_bitcoin_values = []

    for timestep in range(len(list(test_data.values())[0]) -1): # Evaluate over the entire test dataset
        action = agent.get_action(state, noise=False) # No exploration noise during evaluation

        action_set = {}
        action_set['cash'] = action[0]
        action_set['btc'] = action[1]
        action_set['eth'] = action[2]
        action_set['xrp'] = action[3]
        action_set['sol'] = action[4]

        action_set['btc_sl'] = action[5]
        action_set['eth_sl'] = action[6]
        action_set['xrp_sl'] = action[7]
        action_set['sol_sl'] = action[8]

        next_state, _, done, _ = test_env.step(action_set) # Reward is not important for evaluation

        episode_portfolio_values.append(test_env.portfolio_value_history[-1])
        episode_bitcoin_values.append(test_env.bitcoin_value_history[-1])

        state = next_state
        if done:
            break

    # --- Save Evaluation Chart ---
    plt.figure(figsize=(12, 6))
    plt.plot(episode_portfolio_values, label='TD3 Agent Portfolio Value (Test Data)')
    plt.plot(episode_bitcoin_values, label='Hold Bitcoin Portfolio Value (Test Data)', linestyle='--')
    plt.xlabel('Timestep')
    plt.ylabel('Portfolio Value')
    plt.title(f'Episode {episode_num} - Test Data Portfolio Value Comparison') # Updated chart title
    plt.legend()
    chart_path = os.path.join(chart_save_dir, f"test_episode_{episode_num}_chart.png")
    plt.savefig(chart_path)
    plt.close() # Close plot to prevent display and clear memory
    print(f"Saved test data chart to {chart_path}")


if __name__ == '__main__':
    # --- Hyperparameters (Tune these!) ---
    episodes = 1000  # Number of training episodes
    timesteps_per_episode = 730 # Length of each episode (730 hours in a month)
    batch_size = 64  # Batch size for training updates
    buffer_capacity = 100000
    train_actor_every_step = 2 # Delayed actor updates
    sharpe_window = 24 # Window size for Sharpe Ratio calculation
    slippage_factor = 0.0 # Slippage factor (e.g., 0.01 for 1% slippage) # No slippage for now to check convergence
    stop_loss_pct = 0.05 # Stop loss percentage (e.g., 0.05 for 5% drop)
    model_save_interval = 10 # Save model every n episodes

    # --- Directories for saving models and charts ---
    model_save_dir = "saved_models" # Changed directory name to reflect stop loss
    chart_save_dir = "episode_charts" # Changed directory name to reflect stop loss
    os.makedirs(model_save_dir, exist_ok=True) # Create directory if it doesn't exist
    os.makedirs(chart_save_dir, exist_ok=True) # Create directory if it doesn't exist


    # --- Get Data ---
    crypto_data = get_crypto_data()
    train_data, val_data = split_data(crypto_data)
    test_data = val_data # Use val_data as test_data as per the original request

    # --- Environment and Agent Setup ---
    env = CryptoTradingEnvironment(train_data, n_periods=730, sharpe_window=sharpe_window, slippage_factor=slippage_factor, stop_loss_pct=stop_loss_pct, random_start=True)
    current_state_dim = (9,)
    hist_state_dim = (730, 17*len(env.coins),)
    action_dim = len(env.coins) * 2 + 1 # *2 for Stop Loss actions, +1 for cash
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

            action_set['btc_sl'] = action[5]
            action_set['eth_sl'] = action[6]
            action_set['xrp_sl'] = action[7]
            action_set['sol_sl'] = action[8]

            next_state, reward, done, _ = env.step(action_set)

            # print(f'Reward: {reward:4e}')

            episode_rewards.append(reward)
            replay_buffer.record(state, state, action, reward, next_state, next_state, done) # Record experience
            episode_portfolio_values.append(env.portfolio_value_history[-1]) # Record portfolio value
            episode_bitcoin_values.append(env.bitcoin_value_history[-1]) # Record bitcoin value


            if replay_buffer.buffer_counter > batch_size: # Start training after buffer is filled a bit
                actor_loss, critic_loss_1, critic_loss_2 = agent.train_step(replay_buffer, batch_size)
                if agent.total_steps % 10 == 0: # Print loss less frequently
                    print(f"Step {agent.total_steps}, Actor Loss: {actor_loss.numpy():.4e}, Critic Loss 1: {critic_loss_1.numpy():.4e}, Critic Loss 2: {critic_loss_2.numpy():.4e}")


            state = next_state
            if done:
                break

        avg_reward = np.mean(episode_rewards)
        print(f"Episode {episode+1}/{episodes}, Average Reward: {avg_reward:.4f}, Portfolio Value: {env.get_current_portfolio_value():.2f}")

        # --- Save Model Snapshot ---
        if (episode + 1) % model_save_interval == 0:
            agent.actor_model.save(os.path.join(model_save_dir, f"actor_episode_{episode+1}.keras"))
            agent.critic_model_1.save(os.path.join(model_save_dir, f"critic_1_episode_{episode+1}.keras"))
            agent.critic_model_2.save(os.path.join(model_save_dir, f"critic_2_episode_{episode+1}.keras"))
            print(f"Saved model snapshots at episode {episode+1}")

        # --- Evaluate on test data and plot ---
        evaluate_agent(agent, test_data, chart_save_dir, episode+1)


    # --- 5. Evaluation (Placeholder - Implement evaluation on test_data) ---
    # --- Evaluate trained agent on test_data and calculate performance metrics ---
    print("Training finished. Evaluate agent on test data...")
    # --- Implement evaluation loop and metrics calculation here ---